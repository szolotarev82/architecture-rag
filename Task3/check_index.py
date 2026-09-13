#!/usr/bin/env python3
"""check_index.py — Проверка векторного индекса в ChromaDB с выводом в консоль и файл."""

import datetime
from pathlib import Path

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "aeon_nexus_kb"
EMBEDDING_MODEL = "BAAI/bge-m3"
OUTPUT_FILE = Path(__file__).resolve().parent / "index_results.txt"

QUERIES = [
    "Какое оружие использует Ксарн Велгор?",
    "Уязвимости Войд Кора",
    "Кто такой Громмак и какова его роль?",
]


def main():
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    try:
        client.heartbeat()
    except Exception as e:
        print(f"❌ ChromaDB недоступна: {e}")
        exit(1)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print(f"❌ Коллекция '{COLLECTION_NAME}' не найдена. Запустите build_index.py")
        exit(1)

    count = collection.count()
    print(f"✅ Коллекция: {COLLECTION_NAME}, чанков: {count}")
    if count == 0:
        print("⚠️ Индекс пуст. Запустите build_index.py")
        exit(0)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    lines = []
    lines.append("=" * 60)
    lines.append(f"  Проверка индекса")
    lines.append(f"  Коллекция: {COLLECTION_NAME}")
    lines.append(f"  Чанков: {count}")
    lines.append(f"  Модель: {EMBEDDING_MODEL}")
    lines.append(f"  Дата: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("=" * 60)

    for q in QUERIES:
        print(f"\n🔎 «{q}»")
        lines.append(f"\n🔎 Запрос: «{q}»")

        query_vec = embeddings.embed_query(q)
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=3,
            include=["documents", "metadatas", "distances"],
        )

        if not results["documents"][0]:
            print("   Ничего не найдено.")
            lines.append("   Ничего не найдено.")
            continue

        for i, (doc, meta, dist) in enumerate(
            zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            1,
        ):
            source = meta.get("source_name", "?")
            chunk_id = meta.get("chunk_id", "?")
            preview = doc.replace("\n", " ").strip()[:200]

            print(f"   [{i}] {source} (dist={dist:.4f})")
            print(f"       {preview}...")

            lines.append(f"   [{i}] Источник: {source}")
            lines.append(f"       ID: {chunk_id}")
            lines.append(f"       Дистанция: {dist:.4f}")
            lines.append(f"       Фрагмент: {preview}...")
            lines.append("")

    lines.append("=" * 60)
    lines.append("✅ Проверка завершена")

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "=" * 60)
    print(f"✅ Результаты записаны в {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
