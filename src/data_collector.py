import requests
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH = Path("data/raw")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

WIKIPEDIA_TOPICS = {
    "drivers": [
        "Max_Verstappen", "Lewis_Hamilton", "Charles_Leclerc",
        "Lando_Norris", "Carlos_Sainz_Jr.", "Fernando_Alonso",
        "George_Russell", "Sergio_Pérez", "Oscar_Piastri",
        "Sebastian_Vettel", "Nico_Rosberg", "Kimi_Räikkönen",
        "Valtteri_Bottas", "Daniel_Ricciardo", "Lance_Stroll",
        "Esteban_Ocon_(racing_driver)", "Pierre_Gasly", "Yuki_Tsunoda",
        "Mick_Schumacher", "Michael_Schumacher", "Ayrton_Senna",
        "Nigel_Mansell", "Alain_Prost", "Jenson_Button"
    ],
    "teams": [
        "Red_Bull_Racing", "Mercedes-AMG_Petronas_F1_Team",
        "Ferrari_in_Formula_One", "McLaren", "Aston_Martin_F1_Team",
        "Alpine_F1_Team", "Williams_Racing", "Haas_F1_Team",
        "Scuderia_AlphaTauri", "Alfa_Romeo_in_Formula_One"
    ],
    "circuits": [
        "Circuit_de_Monaco", "Silverstone_Circuit", "Monza_Circuit",
        "Circuit_de_Spa-Francorchamps", "Suzuka_International_Racing_Course",
        "Circuit_of_the_Americas", "Bahrain_International_Circuit",
        "Yas_Marina_Circuit", "Circuit_de_Barcelona-Catalunya",
        "Hungaroring", "Autodromo_Enzo_e_Dino_Ferrari",
        "Interlagos", "Albert_Park_Circuit", "Marina_Bay_Street_Circuit"
    ],
    "concepts": [
        "Formula_One", "Drag_reduction_system", "Formula_One_regulations",
        "Formula_One_tyre_compounds", "Pit_stop", "Safety_car",
        "Kinetic_energy_recovery_system", "Formula_One_scoring_system"
    ],
    "seasons": [
        "2018_Formula_One_World_Championship",
        "2019_Formula_One_World_Championship",
        "2020_Formula_One_World_Championship",
        "2021_Formula_One_World_Championship",
        "2022_Formula_One_World_Championship",
        "2023_Formula_One_World_Championship",
        "2024_Formula_One_World_Championship"
    ]
}


# ─────────────────────────────────────────
# SOURCE 1 — JOLPICA (ERGAST) API
# ─────────────────────────────────────────

def fetch_ergast(endpoint: str) -> dict:
    """Fetch a single endpoint from Jolpica API with retry logic."""
    base_url = "https://api.jolpi.ca/ergast/f1"
    url = f"{base_url}/{endpoint}.json?limit=100"

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            time.sleep(5)

    raise Exception(f"Failed to fetch {url} after 3 attempts")


def collect_drivers() -> list[dict]:
    """Fetch unique F1 drivers from 2000 to 2024."""
    print("Collecting drivers...")
    seen = set()
    unique_drivers = []

    for year in range(2000, 2025):
        data = fetch_ergast(f"{year}/drivers")
        drivers = data["MRData"]["DriverTable"]["Drivers"]
        for driver in drivers:
            if driver["driverId"] not in seen:
                seen.add(driver["driverId"])
                driver["season"] = year
                unique_drivers.append(driver)
        time.sleep(0.2)

    print(f"  Found {len(unique_drivers)} unique drivers")
    return unique_drivers


def collect_race_results(years: list[int]) -> list[dict]:
    """Fetch all race results for given years with pagination."""
    print("Collecting race results...")
    all_races = []

    for year in years:
        offset = 0
        limit = 100
        year_races = []

        while True:
            url = f"https://api.jolpi.ca/ergast/f1/{year}/results.json?limit={limit}&offset={offset}"
            for attempt in range(3):
                try:
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    break
                except requests.exceptions.RequestException:
                    time.sleep(5)

            races = data["MRData"]["RaceTable"]["Races"]
            total = int(data["MRData"]["total"])
            year_races.extend(races)
            offset += limit
            time.sleep(0.3)

            if offset >= total:
                break

        all_races.extend(year_races)
        print(f"  {year}: {len(year_races)} races")

    return all_races


def collect_constructors() -> list[dict]:
    """Fetch unique F1 constructors from 2000 to 2024."""
    print("Collecting constructors...")
    seen = set()
    unique = []

    for year in range(2000, 2025):
        data = fetch_ergast(f"{year}/constructors")
        constructors = data["MRData"]["ConstructorTable"]["Constructors"]
        for c in constructors:
            if c["constructorId"] not in seen:
                seen.add(c["constructorId"])
                c["season"] = year
                unique.append(c)
        time.sleep(0.2)

    print(f"  Found {len(unique)} unique constructors")
    return unique


def collect_standings(years: list[int]) -> dict:
    """Fetch driver and constructor championship standings."""
    print("Collecting standings...")
    standings = {"drivers": {}, "constructors": {}}

    for year in years:
        data = fetch_ergast(f"{year}/driverStandings")
        standings_list = data["MRData"]["StandingsTable"]["StandingsLists"]
        if standings_list:
            standings["drivers"][year] = standings_list[0]["DriverStandings"]

        data = fetch_ergast(f"{year}/constructorStandings")
        standings_list = data["MRData"]["StandingsTable"]["StandingsLists"]
        if standings_list:
            standings["constructors"][year] = standings_list[0]["ConstructorStandings"]

        time.sleep(0.2)
        print(f"  {year} standings collected")

    return standings


# ─────────────────────────────────────────
# SOURCE 2 — WIKIPEDIA
# ─────────────────────────────────────────

def fetch_wikipedia(title: str) -> str:
    """
    Fetch full Wikipedia article text.
    Uses the query API with extracts — returns plain text, no HTML.
    Retries up to 3 times on failure.
    Caps at 5000 words to avoid massive articles.
    """
    url = "https://en.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": "F1RAGChatbot/1.0 (educational project; github.com/student)"
    }
    params = {
        "action": "query",
        "titles": title.replace("_", " "),
        "prop": "extracts",
        "explaintext": True,
        "exsectionformat": "plain",
        "redirects": True,
        "format": "json"
    }

    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code != 200:
                time.sleep(3)
                continue

            data = response.json()
            pages = data.get("query", {}).get("pages", {})

            for page_id, page in pages.items():
                if page_id == "-1":
                    return ""
                extract = page.get("extract", "")
                if not extract:
                    time.sleep(3)
                    continue
                words = extract.split()
                if len(words) > 5000:
                    extract = " ".join(words[:5000])
                return extract

        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {e}")
            time.sleep(5)

    return ""

def retry_failed_wikipedia():
    """Retry only the pages that failed in the main collection."""
    failed_pages = {
        "drivers": ["Daniel_Ricciardo", "Esteban_Ocon"],
        "circuits": ["Circuit_de_Monaco", "Silverstone_Circuit",
                     "Albert_Park_Circuit", "Marina_Bay_Street_Circuit"],
        "concepts": ["Formula_One_tyre_compounds", "Formula_One_scoring_system"],
        "seasons": ["2020_Formula_One_World_Championship",
                    "2021_Formula_One_World_Championship"]
    }

    # Load existing wikipedia data
    with open(RAW_DATA_PATH / "wikipedia.json", encoding="utf-8") as f:
        existing = json.load(f)

    recovered = 0
    for category, pages in failed_pages.items():
        if category not in existing:
            existing[category] = {}
        for page in pages:
            print(f"Retrying: {page}...")
            time.sleep(5)  # longer wait before each retry
            text = fetch_wikipedia(page)
            if text:
                existing[category][page] = text
                recovered += 1
                print(f"  ✓ {page} ({len(text.split())} words)")
            else:
                print(f"  ✗ {page} still failing")

    # Save updated file
    with open(RAW_DATA_PATH / "wikipedia.json", "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\nRecovered {recovered}/{sum(len(v) for v in failed_pages.values())} pages")

def collect_wikipedia_data() -> dict:
    """Fetch full Wikipedia articles for all F1 topics."""
    print("Collecting Wikipedia data...")

    results = {}
    total = sum(len(v) for v in WIKIPEDIA_TOPICS.values())
    collected = 0
    failed = []

    for category, pages in WIKIPEDIA_TOPICS.items():
        results[category] = {}
        for page in pages:
            try:
                text = fetch_wikipedia(page)
                if text:
                    results[category][page] = text
                    collected += 1
                    word_count = len(text.split())
                    print(f"  [{collected}/{total}] {page} ✓ ({word_count} words)")
                else:
                    failed.append(page)
                    print(f"  [{collected}/{total}] {page} ✗ empty response")
            except Exception as e:
                failed.append(page)
                print(f"  [{collected}/{total}] {page} ✗ error: {e}")
            time.sleep(2)

    if failed:
        print(f"\n  Failed ({len(failed)}): {', '.join(failed)}")
    print(f"\n  Successfully collected: {collected}/{total} pages")
    return results


# ─────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────

def save_json(data, filename: str):
    """Save data as JSON to data/raw/."""
    path = RAW_DATA_PATH / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    years = list(range(2000, 2025))

    drivers = collect_drivers()
    save_json(drivers, "drivers.json")

    races = collect_race_results(years)
    save_json(races, "race_results.json")

    constructors = collect_constructors()
    save_json(constructors, "constructors.json")

    standings = collect_standings(years)
    save_json(standings, "standings.json")

    wiki_data = collect_wikipedia_data()
    save_json(wiki_data, "wikipedia.json")


if __name__ == "__main__":
    main()