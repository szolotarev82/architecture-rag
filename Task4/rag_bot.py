#!/usr/bin/env python3
"""rag_bot.py — RAG-бот с Few-shot и Chain-of-Thought промптингом.

Pipeline:
  1. Получает текстовый запрос пользователя.
  2. Преобразует в эмбеддинг через BAAI/bge-m3 (тот же энкодер, что и при индексации).
  3. Ищет ближайшие чанки в ChromaDB (top-k retrieval).
  4. Формирует промпт с few-shot примерами + CoT-инструкцией.
  5. Отправляет в LLM (OpenAI-compatible API).
  6. Возвращает ответ с указанием источников.
"""

import os
import sys
import time
import textwrap
from pathlib import Path

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

# ─── Конфигурация ───────────────────────────────────────────────
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "aeon_nexus_kb"
EMBEDDING_MODEL = "BAAI/bge-m3"

# LLM (OpenAI-compatible). По умолчанию — Ollama (локально).
# Можно переключить на любой другой эндпоинт через переменные окружения.
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")

# Параметры RAG
TOP_K = 4
CHUNK_CHAR_LIMIT = 800  # сколько символов чанка включать в контекст

# ─── Few-shot примеры ──────────────────────────────────────────
# Взяты из той же предметной области (вселенная Aeon Nexus).
# Пример 1 основан на конкретных данных из базы знаний.
FEW_SHOT_EXAMPLES = """\
Q: Как называется столица планеты Ти'лора?
A:
Шаги рассуждения:
1. Ищу в предоставленных фрагментах упоминание планеты Ти'лора.
2. Во фрагменте из источника aeon_planets.md сказано: «Столица Ти'лора — город Сайрон, расположенный в экваториальной зоне».
3. Следовательно, ответ — Сайрон.

Ответ: Столица планеты Ти'лора — Сайрон.

---

Q: Какая технология питает HyperRelay?
A:
Шаги рассуждения:
1. Ищу в предоставленных фрагментах упоминание HyperRelay.
2. Во фрагменте из источника aeon_tech.md указано: «HyperRelay работает на основе ядра VoidCore, которое обеспечивает сверхсветовую передачу данных».
3. Следовательно, ответ — VoidCore.

Ответ: HyperRelay питается от ядра VoidCore.
"""

# ─── System-промпт с Chain-of-Thought ────────────────────────
SYSTEM_PROMPT = f"""\
Ты — эксперт-помощник по вселенной Aeon Nexus. Ты отвечаешь на вопросы, \
опираясь исключительно на предоставленные фрагменты базы знаний.

Правила:
1. Сначала размышляй шаг за шагом: пиши, что именно ты ищешь и где находишь это во фрагментах.
2. После рассуждений дай чёткий итоговый ответ в строке «Ответ: ...».
3. Если в предоставленных фрагментах нет информации для ответа, скажи: «Я не знаю.»
4. Не выдумывай факты, которых нет в фрагментах.
5. Указывай источник информации в ответе.

Примеры:

{FEW_SHOT_EXAMPLES}
"""

# ─── RAG-бот ──────────────────────────────────────────────────


class RAGBot:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._init_chroma()
        self._init_embeddings()
        self._init_llm()

    def _init_chroma(self):
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()
        self.collection = client.get_collection(name=COLLECTION_NAME)
        count = self.collection.count()
        if self.verbose:
            print(f"✅ ChromaDB: «{COLLECTION_NAME}», чанков: {count}")

    def _init_embeddings(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        if self.verbose:
            print(f"✅ Энкодер: {EMBEDDING_MODEL}")

    def _init_llm(self):
        self.llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.llm_model = LLM_MODEL
        if self.verbose:
            print(f"✅ LLM: {LLM_MODEL} ({LLM_BASE_URL})")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Векторный поиск ближайших чанков."""
        query_vec = self.embeddings.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append(
                {
                    "text": doc,
                    "source": meta.get("source_name", "?"),
                    "chunk_id": meta.get("chunk_id", "?"),
                    "distance": dist,
                }
            )
        return chunks

    def build_context(self, chunks: list[dict]) -> str:
        """Формирует текстовый блок с найденными фрагментами."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            text = chunk["text"][:CHUNK_CHAR_LIMIT]
            parts.append(
                f"[Фрагмент {i}] (Источник: {chunk['source']}, "
                f"ID: {chunk['chunk_id']}, дистанция: {chunk['distance']:.4f})\n"
                f"{text}"
            )
        return "\n\n---\n\n".join(parts)

    def build_user_message(self, query: str, chunks: list[dict]) -> str:
        """Собирает полный промпт для LLM."""
        context = self.build_context(chunks)
        return f"""\
=== НАЙДЕННЫЕ ФРАГМЕНТЫ ИЗ БАЗЫ ЗНАНИЙ ===
{context}

=== ВОПРОС ПОЛЬЗОВАТЕЛЯ ===
Q: {query}
A:"""

    def generate(self, query: str, chunks: list[dict]) -> str:
        """Вызывает LLM и возвращает текст ответа."""
        user_msg = self.build_user_message(query, chunks)
        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        return response.choices[0].message.content.strip()

    def answer(self, query: str) -> dict:
        """Полный RAG-пайплайн: retrieve → prompt → generate."""
        start = time.time()

        chunks = self.retrieve(query)
        response = self.generate(query, chunks)
        duration = time.time() - start

        return {
            "query": query,
            "response": response,
            "chunks": chunks,
            "duration": duration,
        }


# ─── REPL-интерфейс ──────────────────────────────────────────


def repl():
    """Консольный бот в режиме диалога."""
    print("\n" + "=" * 60)
    print("🤖 RAG-бот — Aeon Nexus Assistant")
    print("=" * 60)
    print("Введите вопрос или 'exit' для выхода.\n")

    bot = RAGBot()

    while True:
        try:
            query = input("\n👤 Вы: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "выход"):
                print("👋 До встречи!")
                break

            result = bot.answer(query)

            print(f"\n🤖 Бот:\n{result['response']}")
            print(f"\n   ⏱ {result['duration']:.2f} сек | "
                  f"источников: {len(result['chunks'])}")
            for ch in result["chunks"]:
                print(f"   📄 {ch['source']} (dist={ch['distance']:.4f})")

        except KeyboardInterrupt:
            print("\n👋 До встречи!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    repl()
