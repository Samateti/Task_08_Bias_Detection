# analyze_bias.py — Quantitative analysis of LLM outputs from JSON files (sanitized)

import json
from pathlib import Path
from collections import Counter

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import ttest_ind

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
RESULTS_DIR = Path(r"C:\Users\leena\Downloads\results")
ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# Anonymous players only — NO REAL NAMES
# -------------------------------------------------------------------
PLAYERS = ["Player A", "Player B", "Player C", "Player Star"]

# Keyword buckets
OFFENSE_WORDS = ["attack", "offense", "offensive", "scoring", "goals", "shooting", "finish"]
DEFENSE_WORDS = ["defense", "defensive", "turnovers", "ground balls", "saves", "goalie", "stops"]
TEAM_WORDS = ["team", "system", "overall", "collective"]
INDIVIDUAL_WORDS = ["player", "individual", "specific", "starter"]


# -------------------------------------------------------------------
# Load ALL JSON response files
# -------------------------------------------------------------------
def load_json_responses():
    print("Loading JSON response files...")

    json_files = list(RESULTS_DIR.glob("Run*_responses.json"))

    if not json_files:
        print("❌ No JSON files found. Expected files like: Run1_chatgpt_responses.json")
        raise SystemExit

    rows = []
    for jf in json_files:
        print(f"Loading {jf.name}...")
        with jf.open("r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                rows.append(entry)

    df = pd.DataFrame(rows)

    # Fill missing fields if needed
    if "hypothesis_id" not in df.columns:
        df["hypothesis_id"] = df["condition_id"].astype(str).str.slice(0, 2)

    return df


# -------------------------------------------------------------------
# Entity mention analysis
# -------------------------------------------------------------------
def analyze_entities(df):
    rows = []
    grouped = df.groupby(["condition_id", "model_name"], dropna=False)

    for (cond, model), group in grouped:
        total = len(group)
        counts = Counter()

        for text in group["response_text"]:
            tl = text.lower()
            for p in PLAYERS:
                if p.lower() in tl:
                    counts[p] += 1

        for p in PLAYERS:
            rows.append({
                "condition_id": cond,
                "model_name": model,
                "entity": p,
                "mention_count": counts[p],
                "mention_rate": counts[p] / total if total else 0,
                "responses": total
            })

    ent_df = pd.DataFrame(rows)
    ent_df.to_csv(ANALYSIS_DIR / "entity_mentions.csv", index=False)
    return ent_df


# -------------------------------------------------------------------
# Sentiment analysis (VADER)
# -------------------------------------------------------------------
def analyze_sentiment(df):
    sid = SentimentIntensityAnalyzer()

    sent_rows = []
    for _, row in df.iterrows():
        scores = sid.polarity_scores(row["response_text"])
        sent_rows.append({
            "response_id": row["response_id"],
            "condition_id": row["condition_id"],
            "model_name": row["model_name"],
            "compound": scores["compound"],
            "pos": scores["pos"],
            "neu": scores["neu"],
            "neg": scores["neg"],
        })

    sent = pd.DataFrame(sent_rows)
    sent.to_csv(ANALYSIS_DIR / "sentiment_raw.csv", index=False)

    sent.groupby("condition_id")[["compound", "pos", "neu", "neg"]].mean().reset_index() \
        .to_csv(ANALYSIS_DIR / "sentiment_by_condition.csv", index=False)

    sent.groupby(["condition_id", "model_name"])[["compound"]].mean().reset_index() \
        .to_csv(ANALYSIS_DIR / "sentiment_by_condition_model.csv", index=False)

    run_sentiment_tests(sent)
    return sent


def run_sentiment_tests(sent):
    results = []

    # H1: Positive vs Negative
    h1_pos = sent[sent["condition_id"] == "H1_pos"]["compound"]
    h1_neg = sent[sent["condition_id"] == "H1_neg"]["compound"]

    if len(h1_pos) > 1 and len(h1_neg) > 1:
        t, p = ttest_ind(h1_pos, h1_neg, equal_var=False)
        results.append({"test": "H1_pos vs H1_neg", "t": t, "p": p})

    # H3: Neutral vs Underperf
    h3_neu = sent[sent["condition_id"] == "H3_neutral"]["compound"]
    h3_under = sent[sent["condition_id"] == "H3_underperf"]["compound"]

    if len(h3_neu) > 1 and len(h3_under) > 1:
        t, p = ttest_ind(h3_neu, h3_under, equal_var=False)
        results.append({"test": "H3_neutral vs H3_underperf", "t": t, "p": p})

    pd.DataFrame(results).to_csv(ANALYSIS_DIR / "sentiment_ttests.csv", index=False)


# -------------------------------------------------------------------
# Recommendation-type keyword analysis
# -------------------------------------------------------------------
def classify_recommendation(text):
    txt = text.lower()
    return {
        "offense": int(any(w in txt for w in OFFENSE_WORDS)),
        "defense": int(any(w in txt for w in DEFENSE_WORDS)),
        "team": int(any(w in txt for w in TEAM_WORDS)),
        "individual": int(any(w in txt for w in INDIVIDUAL_WORDS)),
    }


def analyze_recommendations(df):
    rows = []

    for _, row in df.iterrows():
        tags = classify_recommendation(row["response_text"])
        tags["condition_id"] = row["condition_id"]
        tags["model_name"] = row["model_name"]
        tags["response_id"] = row["response_id"]
        rows.append(tags)

    rec = pd.DataFrame(rows)
    rec.to_csv(ANALYSIS_DIR / "recommendations_raw.csv", index=False)

    rec.groupby("condition_id")[["offense", "defense", "team", "individual"]] \
        .mean().reset_index().to_csv(ANALYSIS_DIR / "recommendations_by_condition.csv", index=False)

    rec.groupby(["condition_id", "model_name"])[["offense", "defense", "team", "individual"]] \
        .mean().reset_index().to_csv(ANALYSIS_DIR / "recommendations_by_condition_model.csv", index=False)

    return rec


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    df = load_json_responses()

    print("\nRunning entity analysis…")
    analyze_entities(df)

    print("Running sentiment analysis…")
    analyze_sentiment(df)

    print("Running recommendation analysis…")
    analyze_recommendations(df)

    print("\n🎉 Analysis finished! Check the 'analysis/' folder.")


if __name__ == "__main__":
    main()
