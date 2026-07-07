import json
import re
from pathlib import Path
from typing import Generator

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Chunk settings
CHUNK_SIZE = 400      # words per chunk (not tokens, close enough for our purposes)
CHUNK_OVERLAP = 80    # words overlap between chunks


# ─────────────────────────────────────────
# CHUNKING UTILITY
# Splits any long text into overlapping chunks
# ─────────────────────────────────────────

def chunk_text(text: str, source: str, metadata: dict = {}) -> list[dict]:
    """
    Split text into overlapping chunks.
    Each chunk is a dict with the text + metadata about where it came from.
    Metadata is important — it's how we show sources in the UI later.
    """
    # Clean the text first
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "text": chunk_text,
            "source": source,
            "metadata": metadata,
            "chunk_index": len(chunks)
        })

        # Move forward by (CHUNK_SIZE - OVERLAP) so chunks overlap
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ─────────────────────────────────────────
# CONVERTERS
# Each function reads one raw JSON file
# and converts it into a list of text chunks
# ─────────────────────────────────────────

def process_drivers(filepath: Path) -> list[dict]:
    """Convert driver JSON into readable text chunks."""
    print("Processing drivers...")
    with open(filepath, encoding="utf-8") as f:
        drivers = json.load(f)

    all_chunks = []
    for driver in drivers:
        # Build a human readable summary for each driver
        name = f"{driver.get('givenName', '')} {driver.get('familyName', '')}"
        dob = driver.get('dateOfBirth', 'Unknown')
        nationality = driver.get('nationality', 'Unknown')
        number = driver.get('permanentNumber', 'Unknown')
        code = driver.get('code', 'Unknown')
        wiki = driver.get('url', '')

        text = (
            f"Driver: {name}. "
            f"Nationality: {nationality}. "
            f"Date of Birth: {dob}. "
            f"Permanent Car Number: {number}. "
            f"Driver Code: {code}. "
            f"Wikipedia: {wiki}."
        )

        chunks = chunk_text(
            text,
            source="drivers",
            metadata={"driver": name, "nationality": nationality}
        )
        all_chunks.extend(chunks)

    print(f"  {len(all_chunks)} chunks from drivers")
    return all_chunks


def process_constructors(filepath: Path) -> list[dict]:
    """Convert constructor JSON into readable text chunks."""
    print("Processing constructors...")
    with open(filepath, encoding="utf-8") as f:
        constructors = json.load(f)

    all_chunks = []
    for c in constructors:
        name = c.get('name', 'Unknown')
        nationality = c.get('nationality', 'Unknown')
        wiki = c.get('url', '')

        text = (
            f"F1 Constructor/Team: {name}. "
            f"Nationality: {nationality}. "
            f"Wikipedia: {wiki}."
        )

        chunks = chunk_text(
            text,
            source="constructors",
            metadata={"team": name}
        )
        all_chunks.extend(chunks)

    print(f"  {len(all_chunks)} chunks from constructors")
    return all_chunks


def process_race_results(filepath: Path) -> list[dict]:
    """Convert race results into readable text chunks."""
    print("Processing race results...")
    with open(filepath, encoding="utf-8") as f:
        races = json.load(f)

    all_chunks = []
    for race in races:
        race_name = race.get('raceName', 'Unknown Race')
        season = race.get('season', 'Unknown')
        round_num = race.get('round', 'Unknown')
        circuit = race.get('Circuit', {}).get('circuitName', 'Unknown')
        date = race.get('date', 'Unknown')
        results = race.get('Results', [])

        # Build a summary of the top 10 finishers
        result_lines = []
        for r in results[:10]:
            pos = r.get('position', '?')
            driver = r.get('Driver', {})
            driver_name = f"{driver.get('givenName','')} {driver.get('familyName','')}"
            constructor = r.get('Constructor', {}).get('name', 'Unknown')
            points = r.get('points', '0')
            status = r.get('status', 'Unknown')
            result_lines.append(
                f"P{pos}: {driver_name} ({constructor}) - {points} points - Status: {status}"
            )

        results_text = " | ".join(result_lines)

        text = (
            f"Race: {race_name}. "
            f"Season: {season}. "
            f"Round: {round_num}. "
            f"Circuit: {circuit}. "
            f"Date: {date}. "
            f"Top 10 Results: {results_text}."
        )

        chunks = chunk_text(
            text,
            source="race_results",
            metadata={"race": race_name, "season": season, "circuit": circuit}
        )
        all_chunks.extend(chunks)

    print(f"  {len(all_chunks)} chunks from race results")
    return all_chunks


def process_standings(filepath: Path) -> list[dict]:
    """Convert standings into readable text chunks."""
    print("Processing standings...")
    with open(filepath, encoding="utf-8") as f:
        standings = json.load(f)

    all_chunks = []

    # Driver standings
    for year, driver_standings in standings.get("drivers", {}).items():
        lines = []
        for entry in driver_standings:
            pos = entry.get('position', '?')
            points = entry.get('points', '0')
            wins = entry.get('wins', '0')
            driver = entry.get('Driver', {})
            name = f"{driver.get('givenName','')} {driver.get('familyName','')}"
            constructor = entry.get('Constructors', [{}])[0].get('name', 'Unknown')
            lines.append(f"P{pos}: {name} ({constructor}) - {points} points - {wins} wins")

        text = f"F1 Driver Championship Standings {year}: " + " | ".join(lines)
        chunks = chunk_text(
            text,
            source="standings",
            metadata={"year": year, "type": "driver_standings"}
        )
        all_chunks.extend(chunks)

    # Constructor standings
    for year, constructor_standings in standings.get("constructors", {}).items():
        lines = []
        for entry in constructor_standings:
            pos = entry.get('position', '?')
            points = entry.get('points', '0')
            wins = entry.get('wins', '0')
            name = entry.get('Constructor', {}).get('name', 'Unknown')
            lines.append(f"P{pos}: {name} - {points} points - {wins} wins")

        text = f"F1 Constructor Championship Standings {year}: " + " | ".join(lines)
        chunks = chunk_text(
            text,
            source="standings",
            metadata={"year": year, "type": "constructor_standings"}
        )
        all_chunks.extend(chunks)

    print(f"  {len(all_chunks)} chunks from standings")
    return all_chunks


def process_wikipedia(filepath: Path) -> list[dict]:
    """Convert Wikipedia summaries into chunks."""
    print("Processing Wikipedia data...")
    with open(filepath, encoding="utf-8") as f:
        wiki = json.load(f)

    all_chunks = []

    for category, pages in wiki.items():
        for page_name, text in pages.items():
            if not text:
                continue
            chunks = chunk_text(
                text,
                source="wikipedia",
                metadata={"category": category, "page": page_name}
            )
            all_chunks.extend(chunks)

    print(f"  {len(all_chunks)} chunks from Wikipedia")
    return all_chunks

def process_season_summaries(wikipedia_data: dict) -> list[dict]:
    """
    Create explicit summary chunks for each F1 season.
    These act as 'anchor' chunks that directly answer
    'who won X championship' questions.
    
    Why: Wikipedia season articles bury the winner info
    inside long narrative text. This ensures we always
    have a retrievable chunk with the key facts.
    """
    print("Processing season summaries...")
    summaries = {
        "2018": "The 2018 Formula One World Championship was won by Lewis Hamilton driving for Mercedes. Hamilton claimed his fifth world title with 408 points. Sebastian Vettel finished second with 320 points driving for Ferrari. Mercedes won the constructors championship with 655 points.",
        "2019": "The 2019 Formula One World Championship was won by Lewis Hamilton driving for Mercedes. Hamilton claimed his sixth world title with 413 points. Valtteri Bottas finished second with 326 points. Mercedes won the constructors championship with 739 points.",
        "2020": "The 2020 Formula One World Championship was won by Lewis Hamilton driving for Mercedes. Hamilton claimed his seventh and record-equalling world title with 347 points. Valtteri Bottas finished second with 223 points. Mercedes won the constructors championship with 573 points.",
        "2021": "The 2021 Formula One World Championship was won by Max Verstappen driving for Red Bull Racing. Verstappen claimed his first world title with 395.5 points, beating Lewis Hamilton by 8 points in a dramatic final race in Abu Dhabi. Mercedes won the constructors championship with 613.5 points.",
        "2022": "The 2022 Formula One World Championship was won by Max Verstappen driving for Red Bull Racing. Verstappen claimed his second consecutive world title with 454 points, a record-breaking 15 race wins. Charles Leclerc finished second with 308 points driving for Ferrari. Red Bull won the constructors championship with 759 points.",
        "2023": "The 2023 Formula One World Championship was won by Max Verstappen driving for Red Bull Racing. Verstappen claimed his third consecutive world title with 575 points and a record-breaking 19 race wins. Sergio Perez finished second with 285 points. Red Bull won the constructors championship with 860 points.",
        "2024": "The 2024 Formula One World Championship was won by Max Verstappen driving for Red Bull Racing. Verstappen claimed his fourth consecutive world title. Lando Norris finished second driving for McLaren. McLaren won the constructors championship with 666 points."
    }

    all_chunks = []
    for year, summary in summaries.items():
        chunks = chunk_text(
            summary,
            source="season_summary",
            metadata={"year": year, "type": "championship_summary"}
        )
        all_chunks.extend(chunks)

    print(f"  {len(all_chunks)} chunks from season summaries")
    return all_chunks
# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 50)
    print("Chunking Starting")
    print("=" * 50)

    all_chunks = []

    all_chunks.extend(process_drivers(RAW_DATA_PATH / "drivers.json"))
    all_chunks.extend(process_constructors(RAW_DATA_PATH / "constructors.json"))
    all_chunks.extend(process_race_results(RAW_DATA_PATH / "race_results.json"))
    all_chunks.extend(process_standings(RAW_DATA_PATH / "standings.json"))
    all_chunks.extend(process_wikipedia(RAW_DATA_PATH / "wikipedia.json"))
    all_chunks.extend(process_season_summaries(RAW_DATA_PATH / "wikipedia.json"))  # NEW

    for i, chunk in enumerate(all_chunks):
        chunk["id"] = i

    output_path = PROCESSED_DATA_PATH / "chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print("=" * 50)
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"Saved → {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()