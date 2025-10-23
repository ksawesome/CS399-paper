"""Quick analysis of question ratio correlation with pedagogical quality."""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Load data
prompt_df = pd.read_csv(DATA_DIR / "prompt_level_metrics.csv")

# Calculate overall correlation
pearson_r, pearson_p = pearsonr(prompt_df["pedagogical_score"], prompt_df["question_ratio"])
spearman_r, spearman_p = spearmanr(prompt_df["pedagogical_score"], prompt_df["question_ratio"])

print(f"Overall Correlation Analysis:")
print(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})")
print(f"  Spearman ρ = {spearman_r:.3f} (p = {spearman_p:.4f})")
print()

# Per-model correlations
print("Per-Model Correlations (Pearson):")
for model in prompt_df["model"].unique():
    model_data = prompt_df[prompt_df["model"] == model]
    r, p = pearsonr(model_data["pedagogical_score"], model_data["question_ratio"])
    print(f"  {model:50s}: r = {r:.3f} (p = {p:.4f})")
print()

# Summary statistics
print("Summary Statistics by Model:")
print(prompt_df.groupby("model")[["pedagogical_score", "question_ratio"]].agg(["mean", "std"]).round(3))
