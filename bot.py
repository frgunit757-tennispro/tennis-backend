"""
Telegram-бот который запускает Mini App
Render Web Service совместимая версия — запускает HTTP сервер на порту
"""

import os
import asyncio
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import Application, CommandHandler, ContextTypes

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8500214628:AAGInqfRQ9Bsn4cZzTLrysyrXFR9gwvMKNc")
MINI_APP_URL = os.getenv("MINI_APP_URL", "https://frgunit757-tennispro.github.io/tennis-backend/index.html")
PORT = int(os.getenv("PORT", 10000))


# ─── Простой HTTP сервер чтобы Render не ругался ───
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Tennis Bot is running!")

    def log_message(self, format, *args):
        pass  # Отключаем логи HTTP


def run_health_server():
    server = HTTPServer(("0.0.0.0", PORT), HealthHandler)
    print(f"Health server запущен на порту {PORT}")
    server.serve_forever()


# ─── Команды бота ───
async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    keyboard = [[
        InlineKeyboardButton(
            "🎾 Открыть Tennis Analyzer",
            web_app=WebAppInfo(url=MINI_APP_URL)
        )
    ]]
    await update.message.reply_text(
        "👋 Привет!\n\n"
        "🎾 *Tennis Analyzer* — анализ матчей ATP\n\n"
        "• Elo-рейтинги по покрытиям\n"
        "• Прогнозы с вероятностями\n"
        "• H2H история игроков\n"
        "• Матчи сегодня с коэффициентами\n\n"
        "Нажми кнопку чтобы открыть приложение 👇",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❓ *Как пользоваться:*\n\n"
        "1. Нажми /start → кнопка 'Открыть'\n"
        "2. Вкладка *Рейтинг* — топ игроков по Elo\n"
        "3. Вкладка *Прогноз* — введи двух игроков и покрытие\n"
        "4. Вкладка *H2H* — история встреч\n"
        "5. Вкладка *Матчи* — сегодняшние матчи с анализом\n\n"
        "⚠️ Прогнозы статистические, не финансовый совет.",
        parse_mode="Markdown"
    )


def main():
    # Запускаем HTTP сервер в отдельном потоке
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()

    # Запускаем бота
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))

    print("Бот запущен (polling)...")
    app.run_polling()


if __name__ == "__main__":
    main()
