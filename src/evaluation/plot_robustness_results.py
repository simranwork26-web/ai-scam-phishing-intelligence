import pandas as pd
import matplotlib.pyplot as plt

INPUT = "results/fusion/robustness_generated/robustness_verified_predictions.csv"
OUTPUT = "results/fusion/robustness_generated/robustness_score_shift.png"

d = pd.read_csv(INPUT)
order = ["professional", "cue_reduced", "combined"]

fig, ax = plt.subplots(figsize=(10, 6))
x = 0
ticks = []
labels = []

for transform in order:
    g = d[d["transformation_type"] == transform].reset_index(drop=True)
    start = x

    for _, row in g.iterrows():
        ax.plot(
            [x, x],
            [row["original_score"], row["rewritten_score"]],
            marker="o",
        )
        x += 1

    ticks.append((start + x - 1) / 2)
    labels.append(transform.replace("_", " ").title())
    x += 2

ax.axhline(0.5, linestyle="--", label="Decision threshold")
ax.set_ylim(0, 1.02)
ax.set_ylabel("DistilBERT phishing probability")
ax.set_title("Robustness to Semantically Preserved Rewriting")
ax.set_xticks(ticks)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT, dpi=200)
print("Saved:", OUTPUT)
