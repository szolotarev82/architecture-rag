#!/usr/bin/env python3
"""demo.py — Демонстрация работы RAG-бота.

Запускает 7 предзаданных диалогов:
  5 — успешные ответы (на основе базы знаний),
  2 — ответы «Я не знаю» (вопросы вне базы знаний).

Результаты сохраняются в файл dialogues.txt.
"""

import datetime
from pathlib import Path

from rag_bot import RAGBot

OUTPUT_FILE = Path(__file__).resolve().parent / "dialogues.txt"

# 5 успешных запросов (ответы должны быть в базе знаний)
SUCCESS_QUERIES = [
    "Какое оружие использует Ксарн Велгор?",
    "Какие уязвимости есть у Войд Кора?",
    "Кто такой Громмак и какова его роль?",
    "Какая технология питает HyperRelay?",
    "Как называется столица планеты Ти'лора?",
]

# 2 запроса, на которые бот должен ответить «Я не знаю»
# (информации нет в базе знаний Aeon Nexus)
FAIL_QUERIES = [
    "Какая сейчас погода в Москве?",
    "Сколько будет 2 + 2?",
]


def main():
    print("🚀 Запуск демо-диалогов...\n")
    bot = RAGBot(verbose=True)

    lines = []
    lines.append("=" * 60)
    lines.append("  ДЕМО-ДИАЛОГИ RAG-БОТА")
    lines.append(f"  Модель: {bot.llm_model}")
    lines.append(f"  Коллекция: {bot.collection.name}")
    lines.append(f"  Чанков: {bot.collection.count()}")
    lines.append(f"  Дата: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("=" * 60)

    # Успешные диалоги
    lines.append("\n" + "─" * 60)
    lines.append("✅ УСПЕШНЫЕ ДИАЛОГИ (5)")
    lines.append("─" * 60)

    for i, q in enumerate(SUCCESS_QUERIES, 1):
        print(f"\n[{i}/{len(SUCCESS_QUERIES)}] {q}")
        result = bot.answer(q)

        lines.append(f"\n{'=' * 60}")
        lines.append(f"ДИАЛОГ #{i} (успешный)")
        lines.append(f"{'=' * 60}")
        lines.append(f"👤 Пользователь: {q}")
        lines.append(f"\n🤖 Бот:\n{result['response']}")
        lines.append(f"\n📋 Источники ({len(result['chunks'])}):")
        for ch in result["chunks"]:
            lines.append(f"   • {ch['source']} (dist={ch['distance']:.4f})")
        lines.append(f"⏱ Время: {result['duration']:.2f} сек")

        print(f"   ✅ Ответ получен ({result['duration']:.2f} сек)")

    # Диалоги с «Я не знаю»
    lines.append("\n\n" + "─" * 60)
    lines.append("❓ ДИАЛОГИ «Я НЕ ЗНАЮ» (2)")
    lines.append("─" * 60)

    for i, q in enumerate(FAIL_QUERIES, 1):
        print(f"\n[{i}/{len(FAIL_QUERIES)}] {q}")
        result = bot.answer(q)

        lines.append(f"\n{'=' * 60}")
        lines.append(f"ДИАЛОГ #{i} (ожидается «Я не знаю»)")
        lines.append(f"{'=' * 60}")
        lines.append(f"👤 Пользователь: {q}")
        lines.append(f"\n🤖 Бот:\n{result['response']}")
        lines.append(f"\n📋 Источники ({len(result['chunks'])}):")
        for ch in result["chunks"]:
            lines.append(f"   • {ch['source']} (dist={ch['distance']:.4f})")
        lines.append(f"⏱ Время: {result['duration']:.2f} сек")

        print(f"   ✅ Ответ получен ({result['duration']:.2f} сек)")

    lines.append("\n" + "=" * 60)
    lines.append("✅ Демо завершено")
    lines.append("=" * 60)

    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Результаты записаны в {OUTPUT_FILE}")
    print(f"   Успешных: {len(SUCCESS_QUERIES)}, «Я не знаю»: {len(FAIL_QUERIES)}")


if __name__ == "__main__":
    main()
