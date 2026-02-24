import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI

# Загрузка переменных окружения
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")

# Инициализация клиента OpenAI для LM Studio
client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")

# Логирование
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот, использующий локальную AI-модель через LM Studio. "
        "Отправь мне сообщение, и я передам его модели."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Просто напиши текст, и я отправлю его в локальную модель для генерации ответа."
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_message = update.message.text
    try:
        # Отправляем запрос к LM Studio
        response = client.chat.completions.create(
            model="local-model",  # Имя модели любое, LM Studio его игнорирует
            messages=[{"role": "user", "content": user_message}],
            temperature=0.7,
            max_tokens=200
        )
        reply = response.choices[0].message.content
    except Exception as e:
        logger.error(f"Ошибка при обращении к LM Studio: {e}")
        reply = "Извините, не удалось получить ответ от модели. Проверьте, запущен ли сервер LM Studio."

    await update.message.reply_text(reply)

def main():
    if not TELEGRAM_TOKEN:
        raise ValueError("Токен Telegram не найден. Укажите его в файле .env")

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Бот запущен...")
    app.run_polling()

if __name__ == "__main__":
    main()