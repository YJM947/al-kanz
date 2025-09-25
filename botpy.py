#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import logging
import logging.handlers
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh = logging.handlers.RotatingFileHandler('bot.log', maxBytes=2*1024*1024, backupCount=2)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(sh)

setup_logging()
logger = logging.getLogger(__name__)

def load_config(path='config.json'):
    with open(path, 'r') as f:
        return json.load(f)

config = load_config()
BOT_TOKEN = config['telegram']['bot_token']
OWNER_CHAT_ID = str(config['telegram']['chat_id'])
TRADING = config['trading']

class DB:
    def __init__(self, path='state.db'):
        self.path = path
        self._init()

    def _init(self):
        with sqlite3.connect(self.path) as con:
            c = con.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS state (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            con.commit()

    @asynccontextmanager
    async def _con(self):
        loop = asyncio.get_running_loop()
        con = await loop.run_in_executor(None, lambda: sqlite3.connect(self.path))
        try:
            yield con
        finally:
            await loop.run_in_executor(None, con.close)

    async def get(self, key, default=None):
        async with self._con() as con:
            cur = con.cursor()
            cur.execute("SELECT value FROM state WHERE key=?", (key,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else default

    async def set(self, key, value):
        async with self._con() as con:
            cur = con.cursor()
            cur.execute("INSERT OR REPLACE INTO state(key,value) VALUES(?,?)", (key, json.dumps(value)))
            con.commit()

class TradingBot:
    def __init__(self):
        self.db = DB()
        self.trading_enabled = False
        self.balance_usd = float(TRADING.get('starting_balance_usd', 50.0))
        self.day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    async def load_state(self):
        self.trading_enabled = bool(await self.db.get('trading_enabled', False))
        self.balance_usd = float(await self.db.get('balance_usd', self.balance_usd))
        self.day_key = await self.db.get('day_key', self.day_key)

    def is_owner(self, update: Update) -> bool:
        return str(update.effective_user.id) == OWNER_CHAT_ID

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_owner(update):
            await update.message.reply_text("⛔ Unauthorized.")
            return
        await self.show_menu(update, context)

    async def cmd_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_owner(update):
            return
        await self.show_menu(update, context)

    async def on_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_owner(update):
            await update.callback_query.answer()
            return
        q = update.callback_query
        await q.answer()
        if q.data == 'toggle':
            self.trading_enabled = not self.trading_enabled
            await self.db.set('trading_enabled', self.trading_enabled)
            await self.show_menu(update, context)
        elif q.data == 'add10':
            self.balance_usd += 10.0
            await self.db.set('balance_usd', self.balance_usd)
            await self.show_menu(update, context)
        elif q.data == 'reset':
            self.balance_usd = float(TRADING.get('starting_balance_usd', 50.0))
            await self.db.set('balance_usd', self.balance_usd)
            await self.show_menu(update, context)
        else:
            await self.show_menu(update, context)

    async def show_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        status = "🟢 ON" if self.trading_enabled else "🔴 OFF"
        text = (
            "🤖 *Trading Bot*\n\n"
            f"💰 Balance: *${self.balance_usd:.2f}*\n"
            f"⚡ Trading: *{status}*\n"
        )
        kb = [
            [InlineKeyboardButton(f"Toggle Trading ({status})", callback_data='toggle')],
            [InlineKeyboardButton("Add $10 (demo)", callback_data='add10')],
            [InlineKeyboardButton("Reset Balance", callback_data='reset')]
        ]
        rm = InlineKeyboardMarkup(kb)
        if update.callback_query:
            await update.callback_query.edit_message_text(text, reply_markup=rm, parse_mode='Markdown')
        else:
            await update.message.reply_text(text, reply_markup=rm, parse_mode='Markdown')

async def main():
    bot = TradingBot()
    await bot.load_state()

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", bot.cmd_start))
    app.add_handler(CommandHandler("menu", bot.cmd_menu))
    app.add_handler(CallbackQueryHandler(bot.on_button))

    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await app.updater.stop()
        await app.stop()

if __name__ == "__main__":
    asyncio.run(main())