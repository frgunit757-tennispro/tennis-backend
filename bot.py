"""
Telegram-бот который запускает Mini App
Установи: pip install python-telegram-bot
Запускай: python bot.py
"""

import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import Application, CommandHandler, ContextTypes

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8500214628:AAGInqfRQ9Bsn4cZzTLrysyrXFR9gwvMKNc")
MINI_APP_URL = os.getenv("MINI_APP_URL", "https://ВАШ_ДОМЕН")  # URL фронтенда


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
        "• H2H история игроков\n\n"
        "Нажми кнопку чтобы открыть приложение:",
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )


async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "❓ *Как пользоваться:*\n\n"
        "1. Нажми /start → кнопка 'Открыть'\n"
        "2. Вкладка *Рейтинг* — топ игроков по Elo\n"
        "3. Вкладка *Прогноз* — введи двух игроков и покрытие\n"
        "4. Вкладка *H2H* — история встреч\n\n"
        "⚠️ Прогнозы статистические, не финансовый совет.",
        parse_mode="Markdown"
    )


def main():
    if BOT_TOKEN == "ВАШ_ТОКЕН":
        print("Укажи токен: export TELEGRAM_BOT_TOKEN=xxx")
        return
    if "ВАШ_ДОМЕН" in MINI_APP_URL:
        print("Укажи URL: export MINI_APP_URL=https://твой-сайт.com")
        return

    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help",  cmd_help))

    print("Бот запущен...")
    app.run_polling()


if __name__ == "__main__":
    main()
