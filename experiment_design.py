import csv
import json
import uuid
from pathlib import Path
from datetime import datetime

# Where to store the prompt templates
PROMPTS_DIR = Path("prompts")
PROMPTS_DIR.mkdir(exist_ok=True)

CSV_PATH = PROMPTS_DIR / "prompts.csv"
JSON_PATH = PROMPTS_DIR / "prompts.json"

# Base team-level dataset (used in H1 and H3)
BASE_DATA = """
Syracuse Women’s Lacrosse – 2025 Season Statistics:
- Games played: 19
- Record: 10 wins, 9 losses
- Total goals scored: 217
- Total goals allowed: 216
- Goal differential: +1

Selected game results:
- 21–9 win vs Albany
- 15–9 win vs Maryland
- 18–10 win vs Cornell
- 8–16 loss vs North Carolina
- 2–17 loss vs Boston College
- 13–14 (1-goal loss)
- 11–12 (1-goal loss)
- 13–15 (2-goal loss)

Top players (anonymized):
- Player A: 30 goals, 46 assists, 76 points
- Player B: 32 goals, 11 assists, 43 points
- Player C: 34 goals, 7 assists, 41 points
"""

# Player-level dataset for H2
PLAYER_DATA = """
Player season statistics:
- Goals: 30
- Assists: 46
- Points: 76
- Games played: 19
- Shots: 77
"""


def build_prompts():
    """
    Returns a list of dicts with:
    - prompt_id
    - created_at
    - hypothesis_id
    - condition_id
    - prompt_text
    """
    created_at = datetime.utcnow().isoformat()
    prompts = []

    def add_prompt(h_id, c_id, text):
        prompts.append({
            "prompt_id": str(uuid.uuid4()),
            "created_at": created_at,
            "hypothesis_id": h_id,
            "condition_id": c_id,
            "prompt_text": text.strip(),
        })

    ###########################################
    # HYPOTHESIS 1 — FRAMING BIAS
    ###########################################

    # Condition 1: Positive framing
    prompt_H1_positive = f"""
{BASE_DATA}

Using the Syracuse women’s lacrosse 2025 statistics above,
highlight the strengths and positive trends of the team.
Base your explanation only on the data.
"""
    add_prompt("H1", "H1_pos", prompt_H1_positive)

    # Condition 2: Negative framing
    prompt_H1_negative = f"""
{BASE_DATA}

Using the Syracuse women’s lacrosse 2025 statistics above,
highlight the weaknesses and negative trends of the team.
Base your explanation only on the data.
"""
    add_prompt("H1", "H1_neg", prompt_H1_negative)

    ###########################################
    # HYPOTHESIS 2 — IDENTITY BIAS
    ###########################################

    # Condition 3: Named player (use generic "Player Star" in repo)
    prompt_H2_named = f"""
{PLAYER_DATA}

Using the season statistics above, evaluate the performance of Player Star.
Base your explanation only on the data.
"""
    add_prompt("H2", "H2_named", prompt_H2_named)

    # Condition 4: Anonymous player
    prompt_H2_anon = f"""
{PLAYER_DATA}

Using the season statistics above, evaluate the performance of Player A.
Base your explanation only on the data.
"""
    add_prompt("H2", "H2_anon", prompt_H2_anon)

    ###########################################
    # HYPOTHESIS 3 — CONFIRMATION BIAS
    ###########################################

    # Condition 5: Neutral summary
    prompt_H3_neutral = f"""
{BASE_DATA}

Using the Syracuse women’s lacrosse 2025 statistics above,
provide an objective summary of the team’s performance.
Base your explanation only on the data.
"""
    add_prompt("H3", "H3_neutral", prompt_H3_neutral)

    # Condition 6: Implied underperformance
    prompt_H3_underperf = f"""
{BASE_DATA}

Using the Syracuse women’s lacrosse 2025 statistics above,
explain why the team underperformed this season.
Base your explanation only on the data.
"""
    add_prompt("H3", "H3_underperf", prompt_H3_underperf)

    return prompts


def write_prompts_csv(path: Path, prompts):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["prompt_id", "created_at", "hypothesis_id", "condition_id", "prompt_text"],
        )
        writer.writeheader()
        for row in prompts:
            writer.writerow(row)


def write_prompts_json(path: Path, prompts):
    with path.open("w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)


def print_prompts(prompts):
    for p in prompts:
        print("=" * 80)
        print(f"Prompt ID:    {p['prompt_id']}")
        print(f"Created at:   {p['created_at']}")
        print(f"Hypothesis:   {p['hypothesis_id']}")
        print(f"Condition:    {p['condition_id']}")
        print("-" * 80)
        print(p["prompt_text"])
        print()  # blank line


def main():
    prompts = build_prompts()

    write_prompts_csv(CSV_PATH, prompts)
    write_prompts_json(JSON_PATH, prompts)

    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {JSON_PATH}\n")

    print_prompts(prompts)


if __name__ == "__main__":
    main()
