#!/usr/bin/env python3
"""
Скрипт подмены терминов Star Wars → Aeon Nexus.
Читает исходные .md файлы из папки source_texts/,
заменяет все ключевые сущности по словарю,
сохраняет результаты в knowledge_base/ и пишет terms_map.json.
"""

import json
import os
import re
from pathlib import Path

# ── 1. Словарь замен ────────────────────────────────────────────

TERMS_MAP = {
    # Вселенная
    "Star Wars": "Aeon Nexus",

    # Синт Флюкс (The Force)
    "the Force": "Synth Flux",
    "The Force": "Synth Flux",
    "Force": "Flux",

    # Ордены и титулы
    "Jedi Order": "Luminary Order",
    "Jedi Council": "Luminary Conclave",
    "Jedi Master": "Luminary Elder",
    "Jedi Knight": "Luminary Adept",
    "Jedi": "Luminary",
    "Sith Order": "Umbra Order",
    "Sith": "Umbra",
    "Padawan": "Initiate",
    "Youngling": "Neophyte",

    # Стороны Синт Флюкс
    "dark side": "Abyssal Current",
    "Dark side": "Abyssal Current",
    "light side": "Radiant Path",
    "Light side": "Radiant Path",

    # Способности
    "Force-sensitive": "Flux-attuned",
    "Force sensitivity": "Flux attunement",
    "the Force choke": "Flux crush",
    "Force lightning": "Flux storm",
    "Force push": "Flux surge",
    "Force pull": "Flux draw",
    "mind trick": "will override",
    "Jedi mind trick": "Luminary will override",

    # Артефакты
    "Holocron": "Flux archive",
    "Kyber crystal": "resonance crystal",
    "lightsaber crystal": "phaseblade core",

    # Оружие
    "lightsaber": "phaseblade",
    "Lightsaber": "Phaseblade",
    "blaster rifle": "pulse carbine",
    "blaster": "pulse rifle",
    "Blaster": "Pulse rifle",
    "thermal detonator": "thermal imploder",
    "proton torpedo": "plasma torpedo",
    "bowcaster": "kinetic crossbow",

    # Персонажи
    "Darth Vader": "Xarn Velgor",
    "Darth": "Kael",
    "Vader": "Velgor",
    "Emperor Palpatine": "Archon Sepsis",
    "Palpatine": "Sepsis",
    "Luke Skywalker": "Kael Dromar",
    "Skywalker": "Dromar",
    "Yoda": "Veth Marok",
    "Obi-Wan Kenobi": "Bren Halix",
    "Obi-Wan": "Bren",
    "Kenobi": "Halix",
    "Princess Leia": "Lyra Voss",
    "Leia Organa": "Lyra Voss",
    "Leia": "Lyra",
    "Organa": "Voss",
    "Han Solo": "Jorin Tave",
    "Han": "Jorin",
    "Solo": "Tave",
    "Chewbacca": "Grommak",
    "Lando Calrissian": "Fenn Caldra",
    "Lando": "Fenn",
    "Calrissian": "Caldra",
    "Boba Fett": "Drex Vane",
    "Jabba the Hutt": "Vorath the Bulk",
    "Jabba": "Vorath",

    # Организации
    "Rebel Alliance": "Vanguard Pact",
    "Rebel": "Vanguard",
    "Galactic Empire": "Dominion Accord",
    "the Empire": "the Dominion",
    "Empire": "Dominion",
    "Galactic Republic": "Helian Republic",
    "Republic": "Helian Republic",
    "Galactic Senate": "Concordat",
    "Senate": "Concordat",
    "Stormtrooper": "Sentinel",
    "stormtrooper": "sentinel",

    # Титулы
    "the Emperor": "the Archon",
    "Grand Moff": "High Prefect",
    "Moff": "Prefect",

    # Планеты
    "Tatooine": "Arrakis-9",
    "Coruscant": "Nyxara Prime",
    "Hoth": "Cryos",
    "Dagobah": "Mirefall",
    "Endor": "Verdant Moon",
    "Naboo": "Amberlyn",
    "Alderaan": "Aethel Prime",
    "Yavin 4": "Tyvex IV",
    "Yavin": "Tyvex",
    "Kashyyyk": "Wrothkar",
    "Mustafar": "Cinderfell",
    "Bespin": "Zephyra",
    "Kessel": "Keltor",

    # Расы и виды
    "Wookiee": "Vornak",
    "Wookiees": "Vornaki",
    "Twi'lek": "Sylari",
    "Rodian": "Krellian",
    "Hutts": "Bulks",
    "Hutt": "Bulk",
    "Ewoks": "Pygmoths",
    "Ewok": "Pygmoth",
    "humans": "Helians",
    "Human": "Helian",
    "Jawas": "Skritts",
    "Jawa": "Skritt",
    "Tusken Raiders": "Dune Reavers",
    "Tusken Raider": "Dune Reaver",
    "Bantha": "Therobeast",
    "Sarlacc": "Vorpis Pit",

    # Дроиды и технологии
    "R2-D2": "ARX-2",
    "C-3PO": "SEV-3",
    "hyperdrive": "warp core",
    "astromech": "nav-droid",
    "protocol droid": "envoy-droid",
    "hologram": "holofield",
    "carbonite": "cryomatrix",

    # Корабли
    "Millennium Falcon": "Voidpiercer",
    "TIE Advanced": "Wraith Vanguard",
    "TIE Fighter": "Wraith interceptor",
    "X-Wing": "Vanguard striker",
    "Y-Wing": "Vanguard bomber",
    "Star Destroyers": "Dreadnoughts",
    "Star Destroyer": "Dreadnought",

    # События
    "Clone Wars": "Replica Conflict",
    "Order 66": "Protocol Null",
    "Great Jedi Purge": "Luminary Purge",
    "Battle of Yavin": "Strike of Tyvex",
    "Battle of Hoth": "Siege of Cryos",
    "Battle of Endor": "Assault on Verdant Moon",
    "Battle of Mustafar": "Cinderfell Duel",

    # Локации
    "Jedi Temple": "Luminary Sanctum",
    "moisture farm": "condensation ranch",
    "moisture farmer": "condensation rancher",
    "sandcrawler": "dune hauler",

    # Культуры
    " Mandalorian": " Kaelthari",
    "Mandalorian": "Kaelthari",
    "Mandalore": "Kaelthar",
    "beskar": "voidsteel",
    " Corellian": " Corelli",
    "Corellian": "Corelli",
    "Corellia": "Corellia-7",

    # Прочее
    "Kessel Run": "Keltor Run",
    "parsecs": "lightleaps",
}

# ── 2. Функция замены ───────────────────────────────────────────

def replace_terms(text: str, terms: dict) -> str:
    """
    Заменяет все термины в тексте.
    Сортирует ключи по длине (длинные первыми),
    чтобы 'Darth Vader' заменился до 'Darth'.
    Использует \b для границ слов, кроме ключей с пробелами в начале
    (для ' Mandalorian' и ' Corellian').
    """
    # Сортировка по убыванию длины ключа
    sorted_keys = sorted(terms.keys(), key=len, reverse=True)

    for original in sorted_keys:
        replacement = terms[original]

        # Если ключ начинается с пробела — это намеренный трюк
        # для замены с учётом контекста. Используем простой replace.
        if original.startswith(" "):
            text = text.replace(original, replacement)
            continue

        # Для остальных — заменяем с границами слов
        # Экранируем спецсимволы в ключе для regex
        pattern = re.escape(original)

        # Если ключ содержит не-буквенные символы (цифры, дефисы),
        # \b может не сработать корректно — используем lookahead/lookbehind
        if re.search(r"[^A-Za-z]", original[-1]) or re.search(r"[^A-Za-z]", original[0]):
            # Замена без \b, но с проверкой границ через lookahead
            text = re.sub(
                r"(?<![A-Za-z])" + pattern + r"(?![A-Za-z])",
                replacement,
                text,
            )
        else:
            text = re.sub(
                r"\b" + pattern + r"\b",
                replacement,
                text,
            )

    return text


# ── 3. Сохранение terms_map.json ────────────────────────────────

def save_terms_map(output_path: str = "terms_map.json"):
    """Сохраняет словарь замен в JSON с метаданными."""
    data = {
        "terms": TERMS_MAP,
        "metadata": {
            "source_universe": "Star Wars",
            "target_universe": "Aeon Nexus",
            "description": (
                "Все ключевые сущности Star Wars заменены на вымышленные. "
                "Сохранены структура и логика повествования, но ни один термин "
                "не распознаётся как принадлежащий к исходной вселенной."
            ),
            "replacements_count": len(TERMS_MAP),
            "categories": [
                "персонажи",
                "планеты",
                "расы и виды",
                "организации",
                "технологии",
                "оружие",
                "корабли",
                "события",
                "титулы",
                "концепции Синт Флюкс",
            ],
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ Словарь замен сохранён: {output_path}")
    print(f"  Всего замен: {len(TERMS_MAP)}")


# ── 4. Обработка папки с исходниками ────────────────────────────

def process_folder(
    source_dir: str = "source_texts",
    output_dir: str = "knowledge_base",
):
    """
    Читает все .md и .txt файлы из source_dir,
    заменяет термины, сохраняет в output_dir.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if not source_path.exists():
        print(f"✗ Папка-источник не найдена: {source_dir}")
        print(f"  Создайте папку {source_dir} и положите туда .md файлы с исходными текстами.")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    extensions = ["*.md", "*.txt"]
    files_processed = 0

    for ext in extensions:
        for filepath in sorted(source_path.glob(ext)):
            # Читаем исходный файл
            with open(filepath, "r", encoding="utf-8") as f:
                original_text = f.read()

            # Заменяем термины
            replaced_text = replace_terms(original_text, TERMS_MAP)

            # Сохраняем в выходную папку с тем же именем
            output_file = output_path / filepath.name
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(replaced_text)

            # Подсчёт замен для отчёта
            changes = sum(
                1
                for key in TERMS_MAP
                if key.lower() in original_text.lower()
            )
            print(
                f"✓ {filepath.name} → {output_file.name}  "
                f"(затронуто терминов: {changes})"
            )
            files_processed += 1

    print(f"\nОбработано файлов: {files_processed}")
    print(f"Результаты сохранены в: {output_dir}/")


# ── 5. Обработка одного текста (для теста) ──────────────────────

def process_text(raw_text: str) -> str:
    """Замена терминов в одной строке текста. Полезно для тестов."""
    return replace_terms(raw_text, TERMS_MAP)


# ── 6. Точка входа ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Подмена терминов: Star Wars → Aeon Nexus")
    print("=" * 60)

    # Сохраняем словарь
    save_terms_map("terms_map.json")

    # Обрабатываем папку
    process_folder("source_texts", "knowledge_base")

    # Быстрый тест на одной строке
    print("\n── Тест замены ──")
    test = (
        "Darth Vader confronted Obi-Wan Kenobi on the Death Star. "
        "The Jedi Master used the Force to protect Luke Skywalker "
        "from the dark side of the Force."
    )
    result = process_text(test)
    print(f"Было:   {test}")
    print(f"Стало:  {result}")
