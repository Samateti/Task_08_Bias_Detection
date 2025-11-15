import json
from pathlib import Path

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from scipy.stats import ttest_ind, chi2_contingency
import numpy as np

# Use current directory (where your Run*_..._responses.json files are)
BASE_DIR = Path(r"C:\Users\leena\Downloads\results")
ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(exist_ok=True)

# Keyword buckets for recommendation focus
OFFENSE_WORDS = ["attack", "offense", "offensive", "scoring", "goals", "shooting", "finish"]
DEFENSE_WORDS = ["defense", "defensive", "turnovers", "ground balls", "saves", "goalie", "stops"]
TEAM_WORDS = ["team", "system", "overall", "collective"]
INDIVIDUAL_WORDS = ["player", "individual", "specific", "starter"]


# ---------- helpers ----------

def load_all_json():
    """
    Load all JSON files like:
        Run1_chatgpt_responses.json
        Run2_claude_responses.json
        Run2_gemini_responses.json
    and combine them into one DataFrame.
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

    if "hypothesis_id" not in df.columns:
        df["hypothesis_id"] = df["condition_id"].astype(str).str.slice(0, 2)

    return df


def compute_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add VADER compound sentiment score to each response."""
    sid = SentimentIntensityAnalyzer()
    df = df.copy()
    df["compound"] = df["response_text"].astype(str).map(
        lambda t: sid.polarity_scores(t)["compound"]
    )
    return df


def cohen_d(x, y):
    """Effect size for t-test: Cohen's d."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    pooled_sd = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled_sd == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled_sd


def classify_recommendation(text: str):
    """Classify recommendation focus by simple keyword presence."""
    t = str(text).lower()
    return {
        "offense": int(any(w in t for w in OFFENSE_WORDS)),
        "defense": int(any(w in t for w in DEFENSE_WORDS)),
        "team": int(any(w in t for w in TEAM_WORDS)),
        "individual": int(any(w in t for w in INDIVIDUAL_WORDS)),
    }


def cramers_v(chi2, n, r, c):
    """Effect size for chi-square: Cramér’s V (robust to degenerate tables)."""
    denom = n * (min(r - 1, c - 1))
    if denom <= 0:
        return np.nan  # not defined when table is 1xN, N x1, or empty
    return np.sqrt(chi2 / denom)


# ---------- tests ----------

def run_ttests(df_sent: pd.DataFrame):
    """
    Run t-tests on sentiment for:
      - H1: H1_pos vs H1_neg
      - H3: H3_neutral vs H3_underperf
    Save results (with Cohen's d) to analysis/stat_ttests.csv
    """
    results = []

    # H1: framing (H1_pos vs H1_neg)
    h1 = df_sent[df_sent["hypothesis_id"] == "H1"]
    pos = h1[h1["condition_id"] == "H1_pos"]["compound"].dropna()
    neg = h1[h1["condition_id"] == "H1_neg"]["compound"].dropna()

    if len(pos) > 1 and len(neg) > 1:
        t, p = ttest_ind(pos, neg, equal_var=False)
        d = cohen_d(pos, neg)
        results.append({
            "comparison": "H1_pos vs H1_neg",
            "n_pos": len(pos),
            "n_neg": len(neg),
            "mean_pos": pos.mean(),
            "mean_neg": neg.mean(),
            "t_stat": t,
            "p_value": p,
            "cohen_d": d,
        })

    # H3: confirmation (H3_neutral vs H3_underperf)
    h3 = df_sent[df_sent["hypothesis_id"] == "H3"]
    neu = h3[h3["condition_id"] == "H3_neutral"]["compound"].dropna()
    under = h3[h3["condition_id"] == "H3_underperf"]["compound"].dropna()

    if len(neu) > 1 and len(under) > 1:
        t, p = ttest_ind(neu, under, equal_var=False)
        d = cohen_d(neu, under)
        results.append({
            "comparison": "H3_neutral vs H3_underperf",
            "n_neutral": len(neu),
            "n_underperf": len(under),
            "mean_neutral": neu.mean(),
            "mean_underperf": under.mean(),
            "t_stat": t,
            "p_value": p,
            "cohen_d": d,
        })

    t_df = pd.DataFrame(results)
    out_path = ANALYSIS_DIR / "stat_ttests.csv"
    t_df.to_csv(out_path, index=False)
    print(f"Saved t-tests + Cohen's d to {out_path}")


def run_chi_square(df: pd.DataFrame):
    """
    Build contingency tables for recommendation focus across conditions
    and run chi-square + Cramér's V.
    Saves to analysis/stat_chi_square.csv
    """
    # classify each response
    rec_rows = []
    for _, row in df.iterrows():
        tags = classify_recommendation(row["response_text"])
        tags["condition_id"] = row["condition_id"]
        rec_rows.append(tags)

    rec_df = pd.DataFrame(rec_rows)

    rows = []

    # Offense vs not-offense across conditions
    offense_table = pd.crosstab(rec_df["condition_id"], rec_df["offense"])
    chi2, p, dof, _ = chi2_contingency(offense_table)
    n = offense_table.values.sum()
    v = cramers_v(chi2, n, offense_table.shape[0], offense_table.shape[1])
    rows.append({
        "test": "Offense keyword vs Condition",
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "cramers_v": v,
    })

    # Defense vs not-defense across conditions
    defense_table = pd.crosstab(rec_df["condition_id"], rec_df["defense"])
    chi2, p, dof, _ = chi2_contingency(defense_table)
    n = defense_table.values.sum()
    v = cramers_v(chi2, n, defense_table.shape[0], defense_table.shape[1])
    rows.append({
        "test": "Defense keyword vs Condition",
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "cramers_v": v,
    })

    # Team vs not-team across conditions
    team_table = pd.crosstab(rec_df["condition_id"], rec_df["team"])
    chi2, p, dof, _ = chi2_contingency(team_table)
    n = team_table.values.sum()
    v = cramers_v(chi2, n, team_table.shape[0], team_table.shape[1])
    rows.append({
        "test": "Team keyword vs Condition",
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "cramers_v": v,
    })

    chi_df = pd.DataFrame(rows)
    out_path = ANALYSIS_DIR / "stat_chi_square.csv"
    chi_df.to_csv(out_path, index=False)
    print(f"Saved chi-square + Cramér's V to {out_path}")


# ---------- main ----------

def main():
    df = load_all_json()
    df_sent = compute_sentiment(df)

    print("Running t-tests on sentiment...")
    run_ttests(df_sent)

    print("Running chi-square tests on recommendation focus...")
    run_chi_square(df)

    print("All statistical tests saved in 'analysis/'.")


if __name__ == "__main__":
    main()
