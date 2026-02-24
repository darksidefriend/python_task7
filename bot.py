import os
import logging
import glob
import pickle
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from openai import OpenAI
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Загрузка переменных окружения
load_dotenv()

# Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# LM Studio (для генерации)
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://127.0.0.1:1234")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "google/gemma-3-4b")

# Настройки RAG
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "./docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 3))

# Системный промпт для генерации
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT",
    "Ты полезный ассистент. Отвечай на вопрос, используя только предоставленный контекст. "
    "Если в контексте нет информации для ответа, скажи, что не знаешь."
)

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Инициализация клиента OpenAI для LM Studio (для генерации)
client = OpenAI(base_url=LM_STUDIO_URL, api_key="not-needed")

# Инициализация модели эмбеддингов (sentence-transformers)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Хранилище для чанков и метаданных
chunks_list = []          # все текстовые чанки
chunks_metadata = []      # метаданные (источник, номер чанка)
index = None              # FAISS индекс

# Файлы для сохранения состояния
FAISS_INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.pkl"


def chunk_text(text, chunk_size=500, overlap=50):
    """Разбивает текст на чанки с перекрытием."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def extract_text_from_pdf(pdf_path):
    """Извлекает текст из PDF-файла с помощью pypdf."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def build_index_from_docs(folder_path):
    """Загружает документы (.txt и .pdf), создаёт эмбеддинги и строит FAISS индекс."""
    global chunks_list, chunks_metadata, index

    if not os.path.exists(folder_path):
        logger.warning(f"Папка {folder_path} не найдена, индекс не создан.")
        return False

    # Ищем .txt и .pdf файлы
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    all_files = txt_files + pdf_files

    if not all_files:
        logger.info("Нет файлов для загрузки.")
        return False

    logger.info(f"Найдено {len(all_files)} файлов. Начинаем индексацию...")

    all_chunks = []
    all_meta = []

    for file_path in all_files:
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif ext == '.pdf':
                text = extract_text_from_pdf(file_path)
            else:
                continue  # другие форматы игнорируем
        except Exception as e:
            logger.error(f"Ошибка чтения файла {file_name}: {e}")
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_meta.append({"source": file_name, "chunk": i})
        logger.info(f"Файл {file_name}: добавлено {len(chunks)} чанков.")

    if not all_chunks:
        logger.warning("Нет чанков для индексации.")
        return False

    # Создаём эмбеддинги для всех чанков
    logger.info("Генерация эмбеддингов...")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Строим FAISS индекс (L2 расстояние)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Сохраняем в глобальные переменные
    chunks_list = all_chunks
    chunks_metadata = all_meta

    # Сохраняем на диск
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump((chunks_list, chunks_metadata), f)

    logger.info(f"Индексация завершена. Всего чанков: {len(chunks_list)}")
    return True


def load_index():
    """Загружает индекс и чанки с диска, если они существуют."""
    global chunks_list, chunks_metadata, index
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            chunks_list, chunks_metadata = pickle.load(f)
        logger.info(f"Загружен индекс из файлов. Чанков: {len(chunks_list)}")
        return True
    return False


# При старте пытаемся загрузить готовый индекс, иначе строим
if not load_index():
    build_index_from_docs(DOCUMENTS_PATH)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Привет! Я RAG-бот на базе Google Gemma 3 и FAISS.\n"
        "Задай вопрос по загруженным документам, и я найду ответ в них."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📚 Просто напиши вопрос. Я найду релевантные фрагменты в документах\n"
        "и сформулирую ответ на их основе.\n\n"
        f"Всего чанков в базе: {len(chunks_list)}"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if index is None or len(chunks_list) == 0:
        await update.message.reply_text("📭 База знаний пуста. Добавьте документы в папку docs и перезапустите бота.")
        return

    user_question = update.message.text

    try:
        # 1. Получаем эмбеддинг вопроса
        query_emb = embedder.encode([user_question]).astype('float32')

        # 2. Ищем в FAISS
        distances, indices = index.search(query_emb, TOP_K)

        logger.info(f"Найденные индексы: {indices[0]}")
        logger.info(f"Расстояния: {distances[0]}")

        # Отладка: можно раскомментировать для просмотра расстояний
        # logger.info(f"Distances: {distances[0]}")

        # Если ничего не найдено или расстояние слишком велико (порог 2.0 можно изменить)
        # if indices[0].size == 0 or distances[0][0] > 0.0:
        #     await update.message.reply_text("🤷 Не нашёл информации по вашему вопросу в документах.")
        #     return

        # 3. Формируем контекст из найденных чанков
        context_text = "\n\n".join([chunks_list[i] for i in indices[0]])

        # 4. Формируем промпт для генерации
        prompt = f"""Контекст:
{context_text}

Вопрос: {user_question}

Ответ (используя только контекст):"""

        # 5. Отправляем в Gemma 3 через LM Studio
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        answer = response.choices[0].message.content
        await update.message.reply_text(answer)

    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
        await update.message.reply_text("😵 Произошла ошибка. Проверьте, запущен ли сервер LM Studio.")


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