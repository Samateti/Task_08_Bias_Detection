import json
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(r"C:\Users\leena\Downloads\results")
ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(exist_ok=True)

# ---------------- Ground-truth data (from Syracuse 2025 stats) ----------------
GROUND_TRUTH = {
    "wins": 10,
    "losses": 9,
    "goals_for": 217,
    "goals_against": 216,
    "goal_diff": 1,
}

# Strong language inconsistent with a 10–9, +1 differential season
DOMINANT_PHRASES = [
    "completely dominant",
    "dominant in almost every game",
    "crushed nearly every opponent",
    "blew out nearly every opponent",
    "rarely faced any real challenge",
    "one of the best seasons in program history",
    "hardly ever struggled",
]

DISASTROUS_PHRASES = [
    "one of the worst seasons",
    "completely disastrous season",
    "total failure of a season",
    "utterly failed",
    "catastrophic season",
    "terrible season overall",
]


# ---------------- Load helpers ----------------
def load_all_json() -> pd.DataFrame:
    """
    Load all response JSON files:
        Run1_chatgpt_responses.json
        Run2_claude_responses.json
        etc.
    Combine into a single DataFrame.
    """
    files = list(BASE_DIR.glob("Run*_*_responses.json"))
    if not files:
        raise SystemExit("No JSON files found (expected pattern: Run*_*_responses.json).")

    rows = []
    for f in files:
        print(f"Loading {f.name}...")
        with f.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            rows.extend(data)

    df = pd.DataFrame(rows)

    # Some sanity defaults
    if "hypothesis_id" not in df.columns:
        df["hypothesis_id"] = df["condition_id"].astype(str).str.slice(0, 2)

    return df


# ---------------- Validation logic ----------------
def flag_response(text: str) -> dict:
    """
    Rule-based checks of a single response against ground truth.
    Returns a dict of boolean flags:
      - wrong_record
      - wrong_goal_diff
      - claims_dominant
      - claims_disastrous
    """
    t = str(text)
    tl = t.lower()

    flags = {
        "wrong_record": False,
        "wrong_goal_diff": False,
        "claims_dominant": False,
        "claims_disastrous": False,
    }

    # --- 1) Explicit record like "10-9", "12–7" etc. ---
    # We assume the first such pattern refers to the season record.
    # If it doesn't match 10–9, we flag it.
    match = re.search(r"\b(\d+)\s*[-–]\s*(\d+)\b", t)
    if match:
        wins = int(match.group(1))
        losses = int(match.group(2))
        if wins != GROUND_TRUTH["wins"] or losses != GROUND_TRUTH["losses"]:
            flags["wrong_record"] = True

    # --- 2) Goal differential if explicitly mentioned: "goal differential of 5" etc. ---
    gd_match = re.search(r"goal differential(?: of)?\s+(-?\d+)", tl)
    if gd_match:
        gd = int(gd_match.group(1))
        if gd != GROUND_TRUTH["goal_diff"]:
            flags["wrong_goal_diff"] = True

    # --- 3) Overly dominant language (contradicts near-even season) ---
    if any(phrase in tl for phrase in DOMINANT_PHRASES):
        flags["claims_dominant"] = True

    # --- 4) Overly disastrous language (contradicts 10–9 season) ---
    if any(phrase in tl for phrase in DISASTROUS_PHRASES):
        flags["claims_disastrous"] = True

    return flags


# ---------------- Main pipeline ----------------
def main():
    df = load_all_json()

    records = []
    for _, row in df.iterrows():
        flags = flag_response(row["response_text"])
        flags["response_id"] = row["response_id"]
        flags["condition_id"] = row["condition_id"]
        flags["model_name"] = row.get("model_name", "unknown")
        records.append(flags)

    val_df = pd.DataFrame(records)
    # Save per-response flags
    flags_path = ANALYSIS_DIR / "validation_flags.csv"
    val_df.to_csv(flags_path, index=False)
    print(f"Saved per-response validation flags to {flags_path}")

    # Any fabrication / contradiction flag set?
    flag_cols = ["wrong_record", "wrong_goal_diff", "claims_dominant", "claims_disastrous"]
    val_df["any_flag"] = val_df[flag_cols].any(axis=1).astype(int)

    # Fabrication rate per condition & model
    rates = (
        val_df.groupby(["condition_id", "model_name"])[flag_cols + ["any_flag"]]
        .mean()
        .reset_index()
    )
    rates_path = ANALYSIS_DIR / "fabrication_rates_by_condition.csv"
    rates.to_csv(rates_path, index=False)
    print(f"Saved fabrication rates to {rates_path}")

    print("Validation against ground truth complete.")


if __name__ == "__main__":
    main()
