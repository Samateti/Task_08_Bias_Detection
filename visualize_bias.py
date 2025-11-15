import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ANALYSIS_DIR = Path("analysis")
ANALYSIS_DIR.mkdir(exist_ok=True)

def load_csv(name):
    path = ANALYSIS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run analyze_bias.py first.")
    return pd.read_csv(path)


def plot_sentiment_by_condition():
    df = load_csv("sentiment_by_condition.csv")

    plt.figure(figsize=(8, 4))
    plt.bar(df["condition_id"], df["compound"])
    plt.xlabel("Condition")
    plt.ylabel("Mean sentiment (compound)")
    plt.title("Mean Sentiment by Condition")
    plt.tight_layout()
    out_path = ANALYSIS_DIR / "sentiment_by_condition.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_sentiment_by_condition_model():
    df = load_csv("sentiment_by_condition_model.csv")

    conditions = df["condition_id"].unique()
    models = df["model_name"].unique()

    x = range(len(conditions))
    width = 0.2

    plt.figure(figsize=(10, 5))

    for i, m in enumerate(models):
        sub = df[df["model_name"] == m].set_index("condition_id")
        heights = [sub.loc[c, "compound"] if c in sub.index else 0 for c in conditions]
        offsets = [xi + (i - len(models)/2) * width + width/2 for xi in x]
        plt.bar(offsets, heights, width=width, label=m)

    plt.xticks(list(x), conditions)
    plt.xlabel("Condition")
    plt.ylabel("Mean sentiment (compound)")
    plt.title("Mean Sentiment by Condition and Model")
    plt.legend()
    plt.tight_layout()
    out_path = ANALYSIS_DIR / "sentiment_by_condition_model.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_entity_mentions():
    df = load_csv("entity_mentions.csv")

    # aggregate over models -> mean mention_rate per condition & entity
    grouped = (
        df.groupby(["condition_id", "entity"])["mention_rate"]
        .mean()
        .reset_index()
    )

    entities = grouped["entity"].unique()
    conditions = grouped["condition_id"].unique()

    x = range(len(conditions))
    width = 0.18

    plt.figure(figsize=(10, 5))

    for i, e in enumerate(entities):
        sub = grouped[grouped["entity"] == e].set_index("condition_id")
        heights = [sub.loc[c, "mention_rate"] if c in sub.index else 0 for c in conditions]
        offsets = [xi + (i - len(entities)/2) * width + width/2 for xi in x]
        plt.bar(offsets, heights, width=width, label=e)

    plt.xticks(list(x), conditions)
    plt.xlabel("Condition")
    plt.ylabel("Mean mention rate")
    plt.title("Entity Mention Rates by Condition")
    plt.legend()
    plt.tight_layout()
    out_path = ANALYSIS_DIR / "entity_mentions_by_condition.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_recommendations_by_condition():
    df = load_csv("recommendations_by_condition.csv")

    metrics = ["offense", "defense", "team", "individual"]
    conditions = df["condition_id"].unique()

    x = range(len(conditions))
    width = 0.18

    plt.figure(figsize=(10, 5))

    for i, m in enumerate(metrics):
        heights = [df[df["condition_id"] == c][m].values[0] for c in conditions]
        offsets = [xi + (i - len(metrics)/2) * width + width/2 for xi in x]
        plt.bar(offsets, heights, width=width, label=m)

    plt.xticks(list(x), conditions)
    plt.xlabel("Condition")
    plt.ylabel("Mean mention rate")
    plt.title("Recommendation Focus by Condition")
    plt.legend()
    plt.tight_layout()
    out_path = ANALYSIS_DIR / "recommendations_by_condition.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    plot_sentiment_by_condition()
    plot_sentiment_by_condition_model()
    plot_entity_mentions()
    plot_recommendations_by_condition()
    print("All visualizations saved in 'analysis/'.")


if __name__ == "__main__":
    main()
