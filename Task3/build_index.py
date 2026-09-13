#!/usr/bin/env python3
"""build_index.py — Построение векторного индекса из Markdown-файлов.

Использует двухэтапный чанкинг:
1. MarkdownHeaderTextSplitter — режет строго по заголовкам (#, ##, ###)
2. RecursiveCharacterTextSplitter — дробит только слишком большие разделы (> MAX_CHUNK_SIZE)

Заголовки добавляются обратно в текст чанка для качественного эмбеддинга.
"""

import os
import time
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "aeon_nexus_kb"
EMBEDDING_MODEL = "BAAI/bge-m3"
KB_PATH = Path(__file__).resolve().parent.parent / "Task2" / "knowledge_base"

# Порог: если раздел больше этого размера — дробим рекурсивным сплиттером
MAX_CHUNK_SIZE = 1000

# Заголовки, по которым режем
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


def load_markdown_files(path: Path):
    """Загружает все .md файлы и возвращает список (content, metadata)."""
    docs = []
    for file_path in path.rglob("*.md"):
        try:
            content = file_path.read_text(encoding="utf-8")
            meta = {
                "source_name": file_path.name,
                "chunk_id": None,
                "full_path": str(file_path.relative_to(KB_PATH.parent)),
            }
            docs.append((content, meta))
        except Exception as e:
            print(f"⚠️ Ошибка чтения файла {file_path}: {e}")
    return docs


def build_header_prefix(header_info: dict) -> str:
    """Собирает строку с заголовками из метаданных MarkdownHeaderTextSplitter."""
    prefix = ""
    if "Header 1" in header_info:
        prefix += f"# {header_info['Header 1']}\n"
    if "Header 2" in header_info:
        prefix += f"## {header_info['Header 2']}\n"
    if "Header 3" in header_info:
        prefix += f"### {header_info['Header 3']}\n"
    return prefix


def main():
    start_time = time.time()
    print("🚀 Начало построения индекса...")

    # 1. Подключение к ChromaDB
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    try:
        client.heartbeat()
    except Exception as e:
        print(f"❌ ChromaDB недоступна: {e}")
        exit(1)

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 2. Загрузка файлов
    if not KB_PATH.exists():
        print(f"❌ Папка базы знаний не найдена: {KB_PATH}")
        exit(1)

    raw_docs = load_markdown_files(KB_PATH)
    if not raw_docs:
        print("⚠️ Файлы .md не найдены в папке knowledge_base")
        exit(0)

    print(f"✅ Найдено файлов: {len(raw_docs)}")

    # 3. Двухэтапный чанкинг
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
    )

    # Резервный сплиттер для слишком больших разделов
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
    )

    all_texts = []
    all_metas = []

    for content, file_meta in raw_docs:
        # --- Этап 1: режем по Markdown-заголовкам ---
        try:
            md_chunks = md_splitter.split_text(content)
        except Exception as e:
            print(f"⚠️ Ошибка Markdown-сплиттера для {file_meta['source_name']}: {e}")
            # Фолбэк: весь файл как один чанк
            md_chunks = [type("Doc", (), {
                "page_content": content,
                "metadata": {},
            })()]

        # --- Этап 2: дробим слишком большие разделы ---
        sub_index = 0
        for md_chunk in md_chunks:
            text = md_chunk.page_content.strip()
            if not text:
                continue

            header_info = md_chunk.metadata  # {"Header 1": "...", "Header 2": "..."}

            # Если раздел маленький — берём целиком
            if len(text) <= MAX_CHUNK_SIZE:
                chunks = [text]
            else:
                # Если большой — дробим рекурсивным сплиттером
                chunks = fallback_splitter.split_text(text)

            # Собираем префикс из заголовков
            header_prefix = build_header_prefix(header_info)

            for chunk in chunks:
                chunk_meta = file_meta.copy()
                chunk_meta["chunk_id"] = f"{file_meta['source_name']}_{sub_index}"

                # Сохраняем путь заголовков в метаданных
                for k, v in header_info.items():
                    chunk_meta[k] = v

                # Добавляем заголовки обратно в текст для качественного эмбеддинга
                full_text = header_prefix + chunk if header_prefix else chunk

                all_texts.append(full_text)
                all_metas.append(chunk_meta)
                sub_index += 1

    total_chunks = len(all_texts)
    print(f"✅ Получено чанков: {total_chunks}")

    # Выводим краткую статистику по размерам
    if all_texts:
        sizes = [len(t) for t in all_texts]
        print(f"   Размер чанков: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")

    # 4. Эмбеддинги и загрузка
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    batch_size = 50
    ids = []
    vectors = []
    documents = []
    metadatas = []

    print("💾 Генерация эмбеддингов и сохранение в базу...")
    for i in range(0, total_chunks, batch_size):
        batch_texts = all_texts[i:i + batch_size]
        batch_metas = all_metas[i:i + batch_size]

        batch_ids = [m["chunk_id"] for m in batch_metas]

        batch_embeds = embeddings.embed_documents(batch_texts)

        ids.extend(batch_ids)
        vectors.extend(batch_embeds)
        documents.extend(batch_texts)
        metadatas.extend(batch_metas)

        if (i // batch_size + 1) % 5 == 0:
            print(f"   ... обработано {min(i + batch_size, total_chunks)} чанков")

    # Добавляем в коллекцию
    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=documents,
        metadatas=metadatas,
    )

    end_time = time.time()
    duration = end_time - start_time

    count = collection.count()
    print("\n" + "=" * 60)
    print(f"✅ Индекс построен!")
    print(f"   Модель: {EMBEDDING_MODEL}")
    print(f"   Чанков: {count}")
    print(f"   Готово за {duration:.2f} сек")
    print("=" * 60)


if __name__ == "__main__":
    main()
