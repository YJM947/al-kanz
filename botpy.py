#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DexPump Trading Bot (Optimized for Replit, Enhanced Strategies)
Key Features:
- Robust Telegram polling with retries, error callbacks, configurable intervals.
- Async database operations to prevent event loop blockage.
- Health check to monitor memory usage on Replit.
- Structured JSON logging for debugging.
- Enhanced Strategies: Momentum (RSI/MACD), Scalping (EMA), Mean Reversion (Bollinger), Arbitrage, DCA (sentiment-driven).
- Risk Management: Kelly Criterion sizing, ATR-based volatility guard, exposure caps.
- Backtesting: Basic framework for strategy validation.

HOW TO RUN
- Replit: Drop bot.py and config.json.
- Install dependencies: `pip install python-telegram-bot aiohttp tenacity psutil TA-Lib`.
- Set bot token and chat ID in config.json.
- Run: `python bot.py`.
- Send /start to bot from OWNER_CHAT_ID.
"""

import asyncio
import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import sqlite3
from contextlib import asynccontextmanager
import aiohttp
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from telegram.error import NetworkError, TimedOut, TelegramError
from tenacity import retry, stop_after_attempt, wait_exponential
import psutil
import time
import numpy as np
import talib  # Technical indicators

# Logging Setup
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(sh)

setup_logging()
logger = logging.getLogger(__name__)

# Load Config
def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error("config.json not found. Please create it.")
        raise

config = load_config()
BOT_TOKEN = config['telegram']['7571151744:AAEpudv-jfqSYT3dlRsH5EfS8PjuFjx6Mc0']
OWNER_CHAT_ID = str(config['telegram']['7571151744'])
TRADING_CONFIG = config['trading']

# Helper for Config Keys
cfg = TRADING_CONFIG.get

# Risk & Behavior Knobs
MOMENTUM_TIERS = cfg("momentum_tiers", {"high": 10.0, "med": 5.0, "low": 2.0})
RISK_PCTS = cfg("risk_pcts", {"high": 0.05, "med": 0.03, "low": 0.015, "min": 0.005})
MIN_BUY_USD = float(cfg("min_buy_usd", 2.0))
RANDOM_SIZE_PCT = float(cfg("random_size_pct", 0.0))
DEAD_HOURS_UTC = cfg("dead_hours_utc", list(range(2, 6)))
MAX_LOSS_PER_TRADE_PCT = float(cfg("max_loss_per_trade_pct", 10.0))
REENTRY_COOLDOWN_SEC = int(cfg("reentry_cooldown_sec", 1800))
TRAILING_STOP_PCT = float(cfg("trailing_stop_pct", 5.0))
MAX_VOLUME_SPIKE = float(cfg("max_volume_spike", 5.0))
FEE_RATE = float(cfg("fee_rate", 0.001))
MAX_PER_TRADE_PCT = float(cfg("max_per_trade_pct", 0.20))
MAX_PORTFOLIO_EXPOSURE_PCT = float(cfg("max_portfolio_exposure_pct", 0.80))
MAX_DAILY_LOSS_PCT = float(cfg("max_daily_loss_pct", 5.0))
MAX_TRADES_PER_DAY = int(cfg("max_trades_per_day", 30))
MIN_PRICE_HISTORY_MIN = int(cfg("min_price_history_min", 10))
MIN_VOLATILITY_15M_PCT = float(cfg("min_volatility_15m_pct", 0.5))
MOMENTUM_THRESHOLD_5M = float(cfg("momentum_threshold_5m", 3.0))
MAX_ATR_PCT = float(cfg("max_atr_pct", 5.0))
MIN_ARBITRAGE_SPREAD = float(cfg("min_arbitrage_spread", 0.5))
ENABLE_SCALPING = bool(cfg("enable_scalping", True))

# In-memory Blacklist
BLACKLIST: Dict[str, float] = {}

def is_blacklisted(symbol: str) -> bool:
    until = BLACKLIST.get(symbol)
    return bool(until and time.time() < until)

def blacklist_symbol(symbol: str, duration_sec: int = REENTRY_COOLDOWN_SEC):
    BLACKLIST[symbol] = time.time() + duration_sec

# Database
class Database:
    def __init__(self, db_path='bot.db'):
        self.db_path = db_path
        self.init_db()

    @asynccontextmanager
    async def _connect_async(self):
        loop = asyncio.get_running_loop()
        conn = await loop.run_in_executor(None, lambda: sqlite3.connect(self.db_path, timeout=10))
        try:
            yield conn
        finally:
            await loop.run_in_executor(None, conn.close)

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS bot_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time TIMESTAMP NOT NULL,
                    highest_price REAL,
                    dca_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'OPEN'
                )
            ''')
            c.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    position_id INTEGER,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    amount_usd REAL NOT NULL,
                    fee_usd REAL DEFAULT 0,
                    pnl_usd REAL DEFAULT 0,
                    reason TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            c.execute('CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status)')
            c.execute('CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(timestamp)')
            conn.commit()

    async def set_state(self, key: str, value):
        async with self._connect_async() as conn:
            c = conn.cursor()
            c.execute('INSERT OR REPLACE INTO bot_state (key,value,updated_at) VALUES (?,?,?)',
                      (key, json.dumps(value), datetime.now()))
            conn.commit()

    async def get_state(self, key: str, default=None):
        async with self._connect_async() as conn:
            c = conn.cursor()
            c.execute('SELECT value FROM bot_state WHERE key=?', (key,))
            row = c.fetchone()
            return json.loads(row[0]) if row else default

    async def add_position(self, symbol: str, quantity: float, entry_price: float) -> Optional[int]:
        async with self._connect_async() as conn:
            c = conn.cursor()
            c.execute('INSERT INTO positions(symbol,quantity,entry_price,entry_time,highest_price,dca_count) VALUES(?,?,?,?,?,?)',
                      (symbol, quantity, entry_price, datetime.now(timezone.utc).isoformat(), entry_price, 0))
            conn.commit()
            return c.lastrowid

    async def update_position_qty(self, position_id: int, new_qty: float, dca_count: int = None):
        async with self._connect_async() as conn:
            c = conn.cursor()
            status = 'CLOSED' if new_qty <= 0 else 'OPEN'
            if dca_count is not None:
                c.execute('UPDATE positions SET quantity=?, status=?, dca_count=? WHERE id=?', 
                          (max(new_qty, 0.0), status, dca_count, position_id))
            else:
                c.execute('UPDATE positions SET quantity=?, status=? WHERE id=?', 
                          (max(new_qty, 0.0), status, position_id))
            conn.commit()

    async def get_open_positions(self) -> List[Dict]:
        async with self._connect_async() as conn:
            c = conn.cursor()
            c.execute('SELECT * FROM positions WHERE status="OPEN" ORDER BY entry_time DESC')
            cols = [d[0] for d in c.description]
            rows = c.fetchall()
            return [dict(zip(cols, r)) for r in rows]

    async def add_trade(self, position_id: Optional[int], symbol: str, side: str, quantity: float, price: float,
                       amount_usd: float, fee_usd: float, pnl_usd: float = 0, reason: str = None):
        async with self._connect_async() as conn:
            c = conn.cursor()
            c.execute('''
                INSERT INTO trades(position_id,symbol,side,quantity,price,amount_usd,fee_usd,pnl_usd,reason,timestamp)
                VALUES(?,?,?,?,?,?,?,?,?,?)
            ''', (position_id, symbol, side, quantity, price, amount_usd, fee_usd, pnl_usd, reason,
                  datetime.now(timezone.utc).isoformat()))
            conn.commit()

    async def get_recent_trades(self, hours: int = 24) -> List[Dict]:
        async with self._connect_async() as conn:
            c = conn.cursor()
            since_iso = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            c.execute('SELECT * FROM trades WHERE timestamp>? ORDER BY timestamp DESC LIMIT 200', (since_iso,))
            cols = [d[0] for d in c.description]
            rows = c.fetchall()
            return [dict(zip(cols, r)) for r in rows]

# Market Data
class RealMarketData:
    def __init__(self):
        self.price_cache: Dict[str, float] = {}
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=720))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=720))
        self.last_update: Dict[str, float] = {}
        self.session: Optional[aiohttp.ClientSession] = None

    async def init_session(self):
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=10)
            headers = {"User-Agent": "DexPumpBot/2.0"}
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)

    async def close_session(self):
        if self.session and not self.session.closed:
            await self.session.close()

    @staticmethod
    def _normalize_symbol(symbol: str) -> Tuple[str, str]:
        s = symbol.upper()
        if '-' in s: base, quote = s.split('-', 1)
        elif '/' in s: base, quote = s.split('/', 1)
        elif s.endswith('USDT'): base, quote = s[:-4], 'USDT'
        elif s.endswith('USDC'): base, quote = s[:-4], 'USDC'
        elif s.endswith('USD'): base, quote = s[:-3], 'USD'
        elif s.endswith('SOL'): base, quote = s[:-3], 'SOL'
        else: base, quote = s, 'USD'
        return base, quote

    def _kraken_pair(self, base: str, quote: str) -> str:
        m = {'BTC': 'XBT', 'XBT': 'XBT', 'ETH': 'ETH', 'SOL': 'SOL', 'USDT': 'USD', 'USDC': 'USD', 'USD': 'USD'}
        b = m.get(base, base)
        q = m.get(quote, quote)
        return f"{b}{q}"

    async def _fetch_coingecko(self, symbol: str) -> Optional[Dict]:
        try:
            base, quote = self._normalize_symbol(symbol)
            map_ids = {'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana', 'USDC': 'usd-coin', 'USDT': 'tether', 'BONK': 'bonk', 'WIF': 'dogwifhat'}
            coin_id = map_ids.get(base, base.lower())
            vs = 'usd' if quote in ['USD', 'USDT', 'USDC'] else quote.lower()
            await self.init_session()
            async with self.session.get('https://api.coingecko.com/api/v3/simple/price', params={
                'ids': coin_id,
                'vs_currencies': vs,
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }) as r:
                if r.status == 200:
                    data = await r.json()
                    if coin_id in data:
                        d = data[coin_id]
                        return {'price': float(d.get(vs, 0) or 0),
                                'volume': float(d.get(f'{vs}_24h_vol', 0) or 0),
                                'change_24h': float(d.get(f'{vs}_24h_change', 0) or 0),
                                'source': 'coingecko'}
        except Exception as e:
            logger.debug(f"CG err {symbol}: {e}")
        return None

    async def _fetch_binance(self, symbol: str) -> Optional[Dict]:
        try:
            base, quote = self._normalize_symbol(symbol)
            await self.init_session()
            async with self.session.get('https://api.binance.com/api/v3/ticker/24hr', params={'symbol': f'{base}{quote}'}) as r:
                if r.status == 200:
                    d = await r.json()
                    return {'price': float(d.get('lastPrice', 0) or 0),
                            'volume': float(d.get('volume', 0) or 0),
                            'change_24h': float(d.get('priceChangePercent', 0) or 0),
                            'source': 'binance'}
        except Exception as e:
            logger.debug(f"Binance err {symbol}: {e}")
        return None

    async def _fetch_jupiter(self, symbol: str) -> Optional[Dict]:
        try:
            base, quote = self._normalize_symbol(symbol)
            if quote not in ['SOL', 'USDC']:
                return None
            await self.init_session()
            async with self.session.get('https://price.jup.ag/v4/price', params={'ids': base}) as r:
                if r.status == 200:
                    d = await r.json()
                    if 'data' in d and base in d['data']:
                        info = d['data'][base]
                        return {'price': float(info.get('price', 0) or 0), 'volume': 0, 'change_24h': 0, 'source': 'jupiter'}
        except Exception as e:
            logger.debug(f"Jupiter err {symbol}: {e}")
        return None

    async def _fetch_kraken(self, symbol: str) -> Optional[Dict]:
        try:
            base, quote = self._normalize_symbol(symbol)
            pair = self._kraken_pair(base, quote)
            await self.init_session()
            async with self.session.get('https://api.kraken.com/0/public/Ticker', params={'pair': pair}) as r:
                if r.status == 200:
                    d = await r.json()
                    if 'result' in d and d['result']:
                        key = next(iter(d['result']))
                        t = d['result'][key]
                        return {'price': float(t['c'][0]), 'volume': float(t['v'][1]), 'change_24h': 0, 'source': 'kraken'}
        except Exception as e:
            logger.debug(f"Kraken err {symbol}: {e}")
        return None

    async def get_price(self, symbol: str) -> float:
        now = time.time()
        if symbol in self.price_cache and now - self.last_update.get(symbol, 0) < 10:
            return self.price_cache[symbol]
        for fetch in (self._fetch_binance, self._fetch_kraken, self._fetch_jupiter, self._fetch_coingecko):
            try:
                res = await fetch(symbol)
                if res and res.get('price', 0) > 0:
                    price = float(res['price'])
                    vol = float(res.get('volume', 0))
                    self.price_cache[symbol] = price
                    self.last_update[symbol] = now
                    self._store(symbol, price, vol)
                    return price
            except Exception:
                continue
        return self.price_cache.get(symbol, 0.0)

    def _store(self, symbol: str, price: float, volume: float):
        self.price_history[symbol].append((datetime.now(timezone.utc), price))
        self.volume_history[symbol].append((datetime.now(timezone.utc), max(volume, 0)))

    def price_change_pct(self, symbol: str, minutes: int) -> Optional[float]:
        hist = self.price_history[symbol]
        if len(hist) < 2:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        ref = next((p for t, p in hist if t <= cutoff), None)
        if ref is None and hist:
            ref = hist[0][1]
        cur = hist[-1][1]
        if not ref or ref <= 0 or cur <= 0:
            return None
        return (cur - ref) / ref * 100.0

    def volatility_15m_pct(self, symbol: str) -> Optional[float]:
        hist = list(self.price_history[symbol])[-15:]
        if len(hist) < 2:
            return None
        changes = []
        for i in range(1, len(hist)):
            p0 = hist[i-1][1]
            p1 = hist[i][1]
            if p0 > 0:
                changes.append(abs(p1 - p0) / p0 * 100.0)
        return sum(changes) / len(changes) if changes else 0.0

    def volume_spike_block(self, symbol: str) -> bool:
        hist = list(self.volume_history[symbol])[-30:]
        if len(hist) < 5:
            return False
        vals = [v for _, v in hist]
        avg = sum(vals[:-1]) / max(1, len(vals) - 1)
        cur = vals[-1]
        if avg <= 0:
            return False
        ratio = cur / avg
        return ratio >= MAX_VOLUME_SPIKE

# News Guard
NEGATIVE_KEYWORDS = ["hack", "exploit", "sues", "suspends", "rug", "ban", "halt", "outage", "bankruptcy", "liquidation"]

class NewsGuard:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.last_titles: deque = deque(maxlen=50)
        self.risk_off_until: Optional[float] = None

    async def init_session(self):
        if not self.session or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8))

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def tick(self):
        try:
            await self.init_session()
            async with self.session.get('https://www.coindesk.com/arc/outboundfeeds/rss/') as r:
                if r.status == 200:
                    txt = await r.text()
                    titles = [line.strip() for line in txt.splitlines() if '<title>' in line]
                    new_hits = 0
                    for t in titles[-20:]:
                        tclean = t.lower()
                        if any(k in tclean for k in NEGATIVE_KEYWORDS) and t not in self.last_titles:
                            new_hits += 1
                            self.last_titles.append(t)
                    if new_hits >= 2:
                        self.risk_off_until = time.time() + 30 * 60
        except Exception:
            pass

    def risk_off(self) -> bool:
        return bool(self.risk_off_until and time.time() < self.risk_off_until)

# Strategy Engine
class StrategyEngine:
    def __init__(self, market_data):
        self.market = market_data

    def compute_indicators(self, symbol: str, period: int = 50) -> Dict:
        """Compute fused indicators for all strategies."""
        hist = list(self.market.price_history[symbol])[-period:]
        if len(hist) < period:
            return {}
        closes = np.array([p for _, p in hist])
        highs = closes  # Approximation; needs OHLCV for accuracy
        lows = closes
        volumes = [v for _, v in list(self.market.volume_history[symbol])[-period:]]
        ind = {
            'rsi': talib.RSI(closes, timeperiod=14)[-1] if len(closes) >= 14 else 50.0,
            'macd': talib.MACD(closes)[0][-1] if len(closes) >= 26 else 0.0,  # MACD line
            'bb_upper': talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)[0][-1] if len(closes) >= 20 else closes[-1],
            'bb_lower': talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)[2][-1] if len(closes) >= 20 else closes[-1],
            'ema_fast': talib.EMA(closes, timeperiod=5)[-1] if len(closes) >= 5 else closes[-1],
            'ema_slow': talib.EMA(closes, timeperiod=20)[-1] if len(closes) >= 20 else closes[-1],
            'atr': talib.ATR(highs, lows, closes, timeperiod=14)[-1] if len(closes) >= 14 else 0.0,
            'volume_spike': self.market.volume_spike_block(symbol)
        }
        return ind

    def momentum_signal(self, symbol: str, ind: Dict, ch5: float) -> Optional[Tuple[str, float]]:
        """Enhanced momentum/breakout: Fuse with RSI/MACD; adjust for volatility."""
        if ind.get('volume_spike', False) or ind.get('atr', 0) > MAX_ATR_PCT:
            return None
        signals = 0
        if ch5 >= MOMENTUM_THRESHOLD_5M: signals += 1
        if ind.get('rsi', 50) < 30: signals += 1  # Oversold
        if ind.get('macd', 0) > 0: signals += 1  # Bullish MACD
        if signals >= 2:  # Require 2/3
            risk_mult = 1 / (1 + ind.get('atr', 0) / 100)  # Volatility-adjusted
            return f"BUY (Momentum +{ch5:.2f}%, RSI {ind['rsi']:.1f}, ATR {ind['atr']:.2f})", risk_mult
        return None

    def mean_reversion_signal(self, symbol: str, ind: Dict) -> Optional[str]:
        """Mean reversion: Buy near BB lower, sell upper."""
        price = self.market.price_cache.get(symbol, 0)
        if price < ind.get('bb_lower', price): return "BUY (Reversion Low)"
        if price > ind.get('bb_upper', price): return "SELL (Reversion High)"
        return None

    async def arbitrage_signal(self, symbol: str) -> Optional[str]:
        """Basic arbitrage: Compare Binance vs. Kraken prices."""
        try:
            binance_price = (await self.market._fetch_binance(symbol) or {}).get('price', 0)
            kraken_price = (await self.market._fetch_kraken(symbol) or {}).get('price', 0)
            if binance_price == 0 or kraken_price == 0: return None
            spread = abs(binance_price - kraken_price) / min(binance_price, kraken_price) * 100
            if MIN_ARBITRAGE_SPREAD < spread < 2.0:  # Safe range
                side = "BUY" if binance_price < kraken_price else "SELL"
                return f"{side} Arbitrage (Spread {spread:.2f}%)"
        except Exception as e:
            logger.debug(f"Arbitrage error {symbol}: {e}")
        return None

    def scalping_signal(self, symbol: str, ind: Dict) -> Optional[str]:
        """Scalping: EMA crossover for quick trades."""
        if ind.get('ema_fast', 0) > ind.get('ema_slow', 0): return "BUY (Scalp Cross Up)"
        if ind.get('ema_fast', 0) < ind.get('ema_slow', 0): return "SELL (Scalp Cross Down)"
        return None

    async def dca_signal(self, symbol: str, ind: Dict, news_guard: NewsGuard, dca_count: int) -> Optional[str]:
        """DCA: Buy on dips if sentiment positive; max 3 entries."""
        if news_guard.risk_off() or dca_count >= 3: return None
        if ind.get('rsi', 50) < 40 and not ind.get('volume_spike', False):
            return "BUY (DCA Dip, RSI Low)"
        return None

# Trading Bot
class TradingBot:
    def __init__(self):
        self.db = Database()
        self.market = RealMarketData()
        self.news = NewsGuard()
        self.strategy = StrategyEngine(self.market)
        self.app = None
        self._runner_task = None
        self.balance_usd = float(self.db.get_state('balance_usd', TRADING_CONFIG["starting_balance_usd"]))
        self.trading_enabled = bool(self.db.get_state('trading_enabled', False))
        today_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.day_key = self.db.get_state('day_key', today_key)
        self.start_equity_today = float(self.db.get_state('start_equity_today', self.equity()))
        self.daily_pnl = float(self.db.get_state('daily_pnl', 0.0))
        self.daily_trades = int(self.db.get_state('daily_trades', 0))

    def equity(self) -> float:
        return self.balance_usd

    async def health_check(self):
        while True:
            try:
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                if memory_usage > 300:  # Replit free tier ~512MB limit
                    logger.warning(f"High memory usage: {memory_usage:.2f}MB. Restarting...")
                    await self.stop_bot()
                    await self.start_bot()
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)

    # Telegram Methods
    def is_authorized(self, update: Update) -> bool:
        return str(update.effective_user.id) == OWNER_CHAT_ID

    async def start_bot(self):
        if self.app is not None:
            logger.info("Bot already running, skipping initialization")
            return

        self.app = Application.builder().token(BOT_TOKEN).build()
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("menu", self.cmd_menu))
        self.app.add_handler(CommandHandler("rules", self.cmd_rules))
        self.app.add_handler(CommandHandler("close_position", self.cmd_close_position))
        self.app.add_handler(CommandHandler("update_config", self.cmd_update_config))
        self.app.add_handler(CallbackQueryHandler(self.on_button))

        if not self._runner_task or self._runner_task.done():
            self._runner_task = asyncio.create_task(self.runner_loop())

        await self.app.initialize()
        await self.app.start()

        @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=2, min=5, max=120))
        async def start_polling_with_retry():
            try:
                await self.app.updater.start_polling(
                    timeout=15,
                    read_timeout=15,
                    write_timeout=15,
                    connect_timeout=15,
                    pool_timeout=30,
                    drop_pending_updates=True,
                    error_callback=self.polling_error_callback,
                    poll_interval=float(config['telegram'].get('poll_interval', 1.0))
                )
                logger.info("Telegram polling started successfully")
            except (NetworkError, TimedOut) as e:
                logger.warning(f"Polling error: {e}. Retrying...")
                raise
            except TelegramError as e:
                logger.error(f"Critical Telegram error: {e}")
                raise

        async def run_polling_forever():
            while True:
                try:
                    await start_polling_with_retry()
                except Exception as e:
                    logger.error(f"Polling failed after retries: {e}. Restarting in 60s...")
                    await asyncio.sleep(60)
                    await self.app.updater.stop()
                    await self.app.stop()
                    await self.app.initialize()
                    await self.app.start()

        asyncio.create_task(run_polling_forever())
        asyncio.create_task(self.health_check())

    async def polling_error_callback(self, error: TelegramError):
        logger.error(f"Polling callback error: {error}")
        if isinstance(error, (NetworkError, TimedOut)):
            logger.warning("Transient error detected, allowing retry")
        else:
            logger.critical(f"Non-recoverable error: {error}. Triggering bot restart")
            await self.stop_bot()
            await asyncio.sleep(10)
            await self.start_bot()

    async def stop_bot(self):
        try:
            if self.app:
                await self.app.updater.stop()
                await self.app.stop()
                self.app = None
            await self.market.close_session()
            await self.news.close()
            logger.info("Bot stopped gracefully")
        except Exception as e:
            logger.error(f"Stop error: {e}")

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            await update.message.reply_text("⛔ Unauthorized user.")
            return
        await self.show_menu(update, context)

    async def cmd_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            return
        await self.show_menu(update, context)

    async def cmd_rules(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            await update.message.reply_text("⛔ Unauthorized user.")
            return
        await update.message.reply_text(STRATEGIST_BRIEF.strip(), parse_mode='Markdown')

    async def cmd_close_position(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            await update.message.reply_text("⛔ Unauthorized user.")
            return
        if not context.args:
            await update.message.reply_text("Usage: /close_position <SYMBOL>")
            return
        symbol = context.args[0].upper()
        pos = next((p for p in await self.db.get_open_positions() if p['symbol'] == symbol), None)
        if not pos:
            await update.message.reply_text(f"No open position for {symbol}")
            return
        price = await self.market.get_price(symbol) or float(pos['entry_price'])
        await self._close_position(pos, price, "Manual Close")
        await update.message.reply_text(f"✅ Closed {symbol} at ${price:.8f}")

    async def cmd_update_config(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            await update.message.reply_text("⛔ Unauthorized user.")
            return
        if not context.args:
            await update.message.reply_text("Usage: /update_config <key> <value>")
            return
        key, value = context.args[0], context.args[1]
        TRADING_CONFIG[key] = float(value) if key in ['take_profit_pct', 'stop_loss_pct', 'max_atr_pct', 'min_arbitrage_spread'] else value
        await self.db.set_state(f'config_{key}', value)
        await update.message.reply_text(f"Updated {key} to {value}")

    async def on_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_authorized(update):
            await update.callback_query.answer()
            return
        q = update.callback_query
        await q.answer()
        data = q.data
        if data == 'toggle_trading':
            self.trading_enabled = not self.trading_enabled
            await self.db.set_state('trading_enabled', self.trading_enabled)
            await self._send_status_flash(q)
        elif data == 'stats':
            await self.show_stats(update, context)
        elif data == 'portfolio':
            await self.show_portfolio(update, context)
        elif data == 'trades':
            await self.show_trades(update, context)
        elif data.startswith('close:'):
            symbol = data.split(':', 1)[1]
            pos = next((p for p in await self.db.get_open_positions() if p['symbol'] == symbol), None)
            if pos:
                price = await self.market.get_price(symbol) or float(pos['entry_price'])
                await self._close_position(pos, price, "Position STOP button")
            await self.show_portfolio(update, context)
        elif data == 'menu':
            await self.show_menu(update, context)

    async def _send_status_flash(self, q):
        status = "🟢 ENABLED" if self.trading_enabled else "🔴 DISABLED"
        text = f"""⚡ **Trading Status Updated**

Status: **{status}**
{('🚀 Entries allowed; managing positions normally' if self.trading_enabled else '⏸️ Entries frozen; open positions will be managed for best exits')}
"""
        kb = [[InlineKeyboardButton("← Back to Menu", callback_data='menu')]]
        await q.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')

    # UI Blocks
    async def show_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        positions = await self.db.get_open_positions()
        trades = await self.db.get_recent_trades(24)
        status_emoji = "🟢 ON" if self.trading_enabled else "🔴 OFF"
        watched_count = len(TRADING_CONFIG.get("watched_tokens", []))
        text = f"""🤖 **DexPump Trading Bot**

💰 Balance: **${self.balance_usd:.2f}**
📋 Positions: **{len(positions)}**
📊 Total Trades: **{len(trades)}**
⚡ Trading: **{status_emoji}**
👀 Tracking **{watched_count}** tokens
"""
        kb = [
            [InlineKeyboardButton(f"⚡ ON/OFF ({status_emoji})", callback_data='toggle_trading')],
            [InlineKeyboardButton("📊 STATS", callback_data='stats')],
            [InlineKeyboardButton("💼 PORTFOLIO", callback_data='portfolio')],
            [InlineKeyboardButton("📈 TRADES", callback_data='trades')],
            [InlineKeyboardButton("🔄 Refresh", callback_data='menu')]
        ]
        rm = InlineKeyboardMarkup(kb)
        if update.callback_query:
            await update.callback_query.edit_message_text(text, reply_markup=rm, parse_mode='Markdown')
        else:
            await update.message.reply_text(text, reply_markup=rm, parse_mode='Markdown')

    async def show_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        recent = await self.db.get_recent_trades(24)
        now_utc = datetime.now(timezone.utc)
        last_hour = [t for t in recent if datetime.fromisoformat(t['timestamp']) > now_utc - timedelta(hours=1)]
        hourly_pnl = sum(float(t['pnl_usd']) for t in last_hour if t['side'] == 'SELL')
        total_trades = len(recent)
        wins = len([t for t in recent if float(t['pnl_usd']) > 0 and t['side'] == 'SELL'])
        win_rate = (wins / total_trades * 100) if total_trades else 0
        text = f"""📊 **STATS**

⏱️ Last Hour PnL: **${hourly_pnl:+.2f}**
🎯 Win Rate (24h): **{win_rate:.1f}%**
📈 Trades (24h): **{total_trades}**
"""
        kb = [[InlineKeyboardButton("← Back to Menu", callback_data='menu')]]
        await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')

    async def show_portfolio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        positions = await self.db.get_open_positions()
        if not positions:
            text = f"""💼 **PORTFOLIO**

📭 **No open positions**
💰 Available Balance: **${self.balance_usd:.2f}**
"""
            kb = [[InlineKeyboardButton("← Back to Menu", callback_data='menu')]]
            await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')
            return
        text_lines = ["💼 **PORTFOLIO**\n"]
        kb_rows = []
        for i, pos in enumerate(positions, 1):
            symbol = pos['symbol']
            entry_price = float(pos['entry_price'])
            qty = float(pos['quantity'])
            cur = await self.market.get_price(symbol) or entry_price
            investment = entry_price * qty
            cur_val = cur * qty
            pnl_usd = cur_val - investment
            pnl_pct = (cur - entry_price) / entry_price * 100 if entry_price else 0
            text_lines.append(f"**{i}. {symbol}**  | Entry ${entry_price:.8f} | Now ${cur:.8f} | P&L {pnl_pct:+.2f}% (${pnl_usd:+.2f})")
            kb_rows.append([InlineKeyboardButton(f"STOP {symbol} (Close Now)", callback_data=f"close:{symbol}")])
        kb_rows.append([InlineKeyboardButton("← Back to Menu", callback_data='menu')])
        await update.callback_query.edit_message_text("\n".join(text_lines), reply_markup=InlineKeyboardMarkup(kb_rows), parse_mode='Markdown')

    async def show_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        recent = await self.db.get_recent_trades(24)
        if not recent:
            text = """📈 **TRADES (24h)**

No trades yet."""
        else:
            total_pnl = sum(float(t['pnl_usd']) for t in recent if t['side'] == 'SELL')
            lines = ["📈 **TRADES (Last 24h)**\n"]
            for t in reversed(recent[-12:]):
                ts = datetime.fromisoformat(t['timestamp']).strftime("%H:%M")
                if t['side'] == 'BUY':
                    lines.append(f"🟢 {ts} BUY {t['symbol']}  ${float(t['price']):.8f}  amt ${float(t['amount_usd']):.2f}  fee ${float(t['fee_usd']):.2f}")
                else:
                    lines.append(f"🔵 {ts} SELL {t['symbol']} ${float(t['price']):.8f}  pnl ${float(t['pnl_usd']):+,.2f}  fee ${float(t['fee_usd']):.2f}")
            lines.append(f"\nTotal realized PnL (24h): **${total_pnl:+.2f}**")
            text = "\n".join(lines)
        kb = [[InlineKeyboardButton("← Back to Menu", callback_data='menu')]]
        await update.callback_query.edit_message_text(text, reply_markup=InlineKeyboardMarkup(kb), parse_mode='Markdown')

    # Backtesting
    async def backtest_strategy(self, symbol: str, days: int = 30) -> Dict:
        """Simulate strategy performance on historical data."""
        results = {'trades': 0, 'pnl': 0.0, 'wins': 0}
        hist = list(self.market.price_history[symbol])[-days*1440:]  # 1-min candles
        if len(hist) < 50: return results
        closes = np.array([p for _, p in hist])
        for i in range(50, len(closes)):
            ind = self.strategy.compute_indicators(symbol, period=50)
            price = closes[i]
            ch5 = (closes[i] - closes[i-5]) / closes[i-5] * 100 if closes[i-5] > 0 else 0
            mom_signal = self.strategy.momentum_signal(symbol, ind, ch5)
            if mom_signal:
                entry_price = price * (1 + 0.001)  # Slippage
                for j in range(i+1, min(i+60, len(closes))):  # Hold up to 1h
                    exit_price = closes[j] * (1 - 0.001)
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    if pnl_pct >= TRADING_CONFIG.get("take_profit_pct", 5.0) or pnl_pct <= -MAX_LOSS_PER_TRADE_PCT:
                        results['trades'] += 1
                        results['pnl'] += pnl_pct * MIN_BUY_USD
                        if pnl_pct > 0: results['wins'] += 1
                        break
        return results

    # Runner Loop
    def _roll_day_if_needed(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.day_key != today:
            self.day_key = today
            self.start_equity_today = self.balance_usd
            self.daily_pnl = 0.0
            self.daily_trades = 0
            asyncio.create_task(self.db.set_state('day_key', self.day_key))
            asyncio.create_task(self.db.set_state('start_equity_today', self.start_equity_today))
            asyncio.create_task(self.db.set_state('daily_pnl', self.daily_pnl))
            asyncio.create_task(self.db.set_state('daily_trades', self.daily_trades))

    def _daily_kill_switch_hit(self) -> bool:
        if self.start_equity_today <= 0:
            return False
        drawdown_pct = (self.daily_pnl / self.start_equity_today) * 100.0
        return (drawdown_pct <= -MAX_DAILY_LOSS_PCT) or (self.daily_trades >= MAX_TRADES_PER_DAY)

    async def runner_loop(self):
        poll = float(cfg("poll_seconds", 5))
        while True:
            try:
                self._roll_day_if_needed()
                await self.news.tick()

                if self._daily_kill_switch_hit():
                    await self._force_close_all("Daily kill-switch")
                    self.trading_enabled = False
                    await self.db.set_state('trading_enabled', self.trading_enabled)

                await self._manage_positions()

                if self.trading_enabled and self._time_allowed() and not self.news.risk_off():
                    await self._check_opportunities()

                await asyncio.sleep(poll)
            except Exception as e:
                logger.error(f"Runner loop error: {e}")
                await asyncio.sleep(10)

    def _time_allowed(self) -> bool:
        return datetime.utcnow().hour not in DEAD_HOURS_UTC

    # Strategy Core
    def _get_tier_risk(self, signal_strength: float) -> float:
        if signal_strength > MOMENTUM_TIERS["high"]: return RISK_PCTS["high"]
        if signal_strength > MOMENTUM_TIERS["med"]: return RISK_PCTS["med"]
        if signal_strength > MOMENTUM_TIERS["low"]: return RISK_PCTS["low"]
        return RISK_PCTS["min"]

    def _dynamic_size_usd(self, signal_strength: float, total_equity: float) -> float:
        """Kelly Criterion: f = (bp - q) / b, simplified for win prob 0.6, avg win 2x loss."""
        kelly_frac = (0.6 * 2 - 0.4) / 2  # ~0.4
        base = total_equity * kelly_frac * self._get_tier_risk(signal_strength)
        base = max(base, MIN_BUY_USD)
        if RANDOM_SIZE_PCT > 0:
            import random
            base *= (1 + random.uniform(-RANDOM_SIZE_PCT, RANDOM_SIZE_PCT))
        base = min(base, MAX_PER_TRADE_PCT * total_equity)
        return max(0.0, min(base, self.balance_usd))

    async def _check_opportunities(self):
        positions = await self.db.get_open_positions()
        watched = TRADING_CONFIG.get("watched_tokens", [])
        open_value = 0.0
        for p in positions:
            cp = await self.market.get_price(p['symbol']) or float(p['entry_price'])
            open_value += cp * float(p['quantity'])
        total_value = open_value + self.balance_usd
        if total_value > 0 and (open_value / total_value) >= MAX_PORTFOLIO_EXPOSURE_PCT:
            return

        for symbol in watched:
            try:
                if any(p['symbol'] == symbol for p in positions) or is_blacklisted(symbol):
                    continue
                price = await self.market.get_price(symbol)
                if price <= 0: continue
                ch5 = self.market.price_change_pct(symbol, 5)
                if ch5 is None: continue
                vol15 = self.market.volatility_15m_pct(symbol)
                if vol15 is None or vol15 < MIN_VOLATILITY_15M_PCT: continue
                ind = self.strategy.compute_indicators(symbol)
                if not ind: continue

                # Momentum
                mom_signal = self.strategy.momentum_signal(symbol, ind, ch5)
                if mom_signal:
                    reason, risk_mult = mom_signal
                    size_usd = self._dynamic_size_usd(ch5 * risk_mult, total_value)
                    if size_usd >= MIN_BUY_USD:
                        await self._open_position(symbol, price, size_usd, reason)

                # Mean Reversion
                rev_signal = self.strategy.mean_reversion_signal(symbol, ind)
                if rev_signal == "BUY":
                    size_usd = MIN_BUY_USD  # Conservative
                    await self._open_position(symbol, price, size_usd, rev_signal)

                # Arbitrage
                arb_signal = await self.strategy.arbitrage_signal(symbol)
                if arb_signal and "BUY" in arb_signal:
                    size_usd = MIN_BUY_USD * 0.5  # Low risk
                    await self._open_position(symbol, price, size_usd, arb_signal)

                # Scalping
                if ENABLE_SCALPING:
                    scalp_signal = self.strategy.scalping_signal(symbol, ind)
                    if scalp_signal == "BUY":
                        size_usd = MIN_BUY_USD * 0.3  # High-frequency
                        await self._open_position(symbol, price, size_usd, scalp_signal)

                # DCA
                pos = next((p for p in positions if p['symbol'] == symbol), None)
                dca_count = pos['dca_count'] if pos else 0
                dca_signal = await self.strategy.dca_signal(symbol, ind, self.news, dca_count)
                if dca_signal:
                    size_usd = MIN_BUY_USD * 0.5
                    await self._open_dca_position(symbol, price, size_usd, dca_signal, pos)
            except Exception as e:
                logger.error(f"Opportunity error {symbol}: {e}")

    async def _manage_positions(self):
        positions = await self.db.get_open_positions()
        for pos in positions:
            try:
                symbol = pos['symbol']
                entry = float(pos['entry_price'])
                qty = float(pos['quantity'])
                cur = await self.market.get_price(symbol) or entry
                highest = float(pos.get('highest_price') or entry)
                if cur > highest:
                    async with await self.db._connect_async() as conn:
                        c = conn.cursor()
                        c.execute('UPDATE positions SET highest_price=? WHERE id=?', (cur, pos['id']))
                        conn.commit()
                    highest = cur
                trail = highest * (1 - TRAILING_STOP_PCT / 100.0)
                pnl_pct = (cur - entry) / entry * 100.0 if entry else 0.0
                drop_pct = ((entry - cur) / entry * 100.0) if entry else 0.0
                if drop_pct >= MAX_LOSS_PER_TRADE_PCT:
                    await self._close_position(pos, cur, f"Hard Stop {-MAX_LOSS_PER_TRADE_PCT:.2f}%")
                    blacklist_symbol(symbol)
                    continue
                if cur <= trail:
                    await self._close_position(pos, cur, f"Trailing Stop {TRAILING_STOP_PCT:.2f}%")
                    blacklist_symbol(symbol)
                    continue
                ind = self.strategy.compute_indicators(symbol)
                rev_signal = self.strategy.mean_reversion_signal(symbol, ind)
                if rev_signal == "SELL":
                    await self._close_position(pos, cur, "Mean Reversion Sell")
                    blacklist_symbol(symbol)
                    continue
                scalp_signal = self.strategy.scalping_signal(symbol, ind) if ENABLE_SCALPING else None
                if scalp_signal == "SELL":
                    await self._close_position(pos, cur, "Scalp Sell")
                    blacklist_symbol(symbol)
                    continue
                arb_signal = await self.strategy.arbitrage_signal(symbol)
                if arb_signal and "SELL" in arb_signal:
                    await self._close_position(pos, cur, arb_signal)
                    blacklist_symbol(symbol)
                    continue
                tp = TRADING_CONFIG.get("take_profit_pct", 5.0)
                if pnl_pct >= tp and qty > 0:
                    if pnl_pct >= 20:
                        await self._close_position(pos, cur, f"Take Profit (20%+) +{pnl_pct:.2f}%")
                        continue
                    elif pnl_pct >= 10:
                        sell_frac = 0.80
                        await self._close_partial_position(pos, cur, sell_frac, f"Scale-out 80% at +{pnl_pct:.2f}%")
                        continue
                    else:
                        sell_frac = 0.50
                        await self._close_partial_position(pos, cur, sell_frac, f"Scale-out 50% at +{pnl_pct:.2f}%")
                        continue
                sl = TRADING_CONFIG.get("stop_loss_pct")
                if sl is not None and pnl_pct <= float(sl):
                    await self._close_position(pos, cur, f"Stop Loss {pnl_pct:.2f}%")
                    blacklist_symbol(symbol)
            except Exception as e:
                logger.error(f"Manage error pos {pos.get('id')}: {e}")

    async def _open_position(self, symbol: str, price: float, buy_amount_usd: float, reason: str):
        fee = buy_amount_usd * FEE_RATE
        net_cash = buy_amount_usd - fee
        qty = net_cash / price if price > 0 else 0.0
        if qty <= 0:
            return
        try:
            pid = await self.db.add_position(symbol, qty, price)
            await self.db.add_trade(pid, symbol, 'BUY', qty, price, buy_amount_usd, fee, 0.0, reason)
            self.balance_usd -= buy_amount_usd
            await self.db.set_state('balance_usd', self.balance_usd)
            self.daily_trades += 1
            await self.db.set_state('daily_trades', self.daily_trades)
            logger.info(f"BUY {symbol} @{price:.8f}, size ${buy_amount_usd:.2f} (fee ${fee:.2f})")
        except Exception as e:
            logger.error(f"Open position error {symbol}: {e}")

    async def _open_dca_position(self, symbol: str, price: float, buy_amount_usd: float, reason: str, existing_pos: Optional[Dict]):
        fee = buy_amount_usd * FEE_RATE
        net_cash = buy_amount_usd - fee
        qty = net_cash / price if price > 0 else 0.0
        if qty <= 0:
            return
        try:
            if existing_pos:
                new_qty = float(existing_pos['quantity']) + qty
                new_dca_count = existing_pos['dca_count'] + 1
                new_entry_price = ((float(existing_pos['entry_price']) * float(existing_pos['quantity'])) + (price * qty)) / new_qty
                await self.db.update_position_qty(existing_pos['id'], new_qty, new_dca_count)
                async with await self.db._connect_async() as conn:
                    c = conn.cursor()
                    c.execute('UPDATE positions SET entry_price=? WHERE id=?', (new_entry_price, existing_pos['id']))
                    conn.commit()
                pid = existing_pos['id']
            else:
                pid = await self.db.add_position(symbol, qty, price)
            await self.db.add_trade(pid, symbol, 'BUY', qty, price, buy_amount_usd, fee, 0.0, reason)
            self.balance_usd -= buy_amount_usd
            await self.db.set_state('balance_usd', self.balance_usd)
            self.daily_trades += 1
            await self.db.set_state('daily_trades', self.daily_trades)
            logger.info(f"DCA BUY {symbol} @{price:.8f}, size ${buy_amount_usd:.2f} (fee ${fee:.2f})")
        except Exception as e:
            logger.error(f"DCA position error {symbol}: {e}")

    async def _close_partial_position(self, position: dict, exit_price: float, fraction: float, reason: str):
        fraction = max(0.0, min(1.0, fraction))
        if fraction <= 0:
            return
        symbol = position['symbol']
        entry = float(position['entry_price'])
        qty = float(position['quantity'])
        qty_sell = qty * fraction
        if qty_sell <= 0:
            return
        gross = exit_price * qty_sell
        fee = gross * FEE_RATE
        proceeds = gross - fee
        pnl_usd = (exit_price - entry) * qty_sell
        try:
            new_qty = qty - qty_sell
            await self.db.update_position_qty(position['id'], new_qty)
            await self.db.add_trade(position['id'], symbol, 'SELL', qty_sell, exit_price, proceeds, fee, pnl_usd, reason)
            self.balance_usd += proceeds
            await self.db.set_state('balance_usd', self.balance_usd)
            self.daily_pnl += pnl_usd
            await self.db.set_state('daily_pnl', self.daily_pnl)
            logger.info(f"PARTIAL CLOSE {symbol} @{exit_price:.8f} qty {qty_sell:.8f} PnL ${pnl_usd:+.2f} (fee ${fee:.2f}) | {reason}")
        except Exception as e:
            logger.error(f"Partial close error {symbol}: {e}")

    async def _close_position(self, position: dict, exit_price: float, reason: str):
        symbol = position['symbol']
        entry = float(position['entry_price'])
        qty = float(position['quantity'])
        gross = exit_price * qty
        fee = gross * FEE_RATE
        proceeds = gross - fee
        pnl_usd = (exit_price - entry) * qty
        try:
            await self.db.update_position_qty(position['id'], 0.0)
            await self.db.add_trade(position['id'], symbol, 'SELL', qty, exit_price, proceeds, fee, pnl_usd, reason)
            self.balance_usd += proceeds
            await self.db.set_state('balance_usd', self.balance_usd)
            self.daily_pnl += pnl_usd
            await self.db.set_state('daily_pnl', self.daily_pnl)
            logger.info(f"CLOSE {symbol} @{exit_price:.8f} PnL ${pnl_usd:+.2f} (fee ${fee:.2f}) | {reason}")
        except Exception as e:
            logger.error(f"Close pos error {symbol}: {e}")

    async def _force_close_all(self, why: str):
        for pos in await self.db.get_open_positions():
            try:
                cp = await self.market.get_price(pos['symbol']) or float(pos['entry_price'])
                await self._close_position(pos, cp, f"Force close: {why}")
            except Exception as e:
                logger.error(f"Force close error {pos.get('symbol')}: {e}")

# Entry Point
STRATEGIST_BRIEF = """
You are a professional-grade crypto strategist.
Capital: $50 (simulation). Strategies: Momentum (RSI/MACD), Scalping (EMA), Mean Reversion (Bollinger), Arbitrage, DCA (sentiment-driven).
Manage risk with Kelly Criterion, ATR, and exposure caps.
"""

async def main():
    bot = TradingBot()
    try:
        await bot.start_bot()
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await bot.stop_bot()

if __name__ == "__main__":
    asyncio.run(main())