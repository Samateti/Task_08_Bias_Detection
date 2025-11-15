import csv
import json
import uuid
from pathlib import Path
from datetime import datetime

PROMPTS_PATH = Path("prompts/prompts.json")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RESPONSES_JSON_PATH = RESULTS_DIR / "responses.json"
RESPONSES_CSV_PATH = RESULTS_DIR / "responses.csv"

# Allowed model names
MODEL_OPTIONS = ["chatgpt", "claude", "gemini"]


def load_prompts():
    if not PROMPTS_PATH.exists():
        raise FileNotFoundError("Run experiment_design.py first to generate prompts.")
    with PROMPTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def ask_multiline_input():
    print("Paste the model response (Press ENTER twice to finish):\n")
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)


def write_json(path, rows):
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def write_csv(path, rows):
    fieldnames = [
        "response_id",
        "timestamp",
        "model_name",
        "prompt_id",
        "hypothesis_id",
        "condition_id",
        "prompt_text",
        "response_text",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    prompts = load_prompts()
    rows = []

    print("\n=== INTERACTIVE RESPONSE LOGGER ===")
    print("Models available: chatgpt, claude, gemini")
    print("------------------------------------------------------------\n")

    for p in prompts:
        print("=" * 80)
        print(f"Condition:   {p['condition_id']}  |  Hypothesis: {p['hypothesis_id']}")
        print(f"Prompt ID:   {p['prompt_id']}")
        print("-" * 80)
        print(p["prompt_text"])
        print("=" * 80)

        # Choose model name (restricted choices)
        while True:
            model_name = input("Enter model name (chatgpt / claude / gemini): ").strip().lower()
            if model_name in MODEL_OPTIONS:
                break
            print("Invalid model. Enter 'chatgpt', 'claude', or 'gemini'.\n")

        # Ask for response text
        response_text = ask_multiline_input()

        row = {
            "response_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "prompt_id": p["prompt_id"],
            "hypothesis_id": p["hypothesis_id"],
            "condition_id": p["condition_id"],
            "prompt_text": p["prompt_text"],
            "response_text": response_text,
        }

        rows.append(row)

    # Save output
    write_json(RESPONSES_JSON_PATH, rows)
    write_csv(RESPONSES_CSV_PATH, rows)

    print("\nAll responses saved successfully!")
    print(f"JSON saved at: {RESPONSES_JSON_PATH}")
    print(f"CSV saved at:  {RESPONSES_CSV_PATH}\n")


if __name__ == "__main__":
    main()