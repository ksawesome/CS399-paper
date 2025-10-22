"""Recompute statistical analyses and generate supplementary figures/tables."""
from __future__ import annotations

from pathlib import Path
from itertools import combinations
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, chi2, rankdata, binomtest
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Canonical model name mapping for presentation
NAME_MAP = {
    "openai-gpt-4o-mini": "GPT-4o mini",
    "anthropic-claude-3-sonnet-20240229": "Claude 3 Sonnet",
    "google-gemini-2.5-flash": "Gemini 2.5 Flash",
    "cohere-command-r-08-2024": "Command R 08-2024",
    "meta-llama-Llama-4-Maverick-17B-128E-Instruct": "Llama 4 Maverick 17B",
}
MODELS_ORDER = list(NAME_MAP.keys())
RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------
def _pivot_metric(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot prompt-level metrics to prompt x model matrix."""
    matrix = (
        df.pivot_table(index="prompt_id", columns="model", values=value)
        .sort_index(axis=1)
        .dropna()
    )
    return matrix


def _rank_biserial(diff: np.ndarray) -> float:
    """Compute rank-biserial correlation for paired differences."""
    diff = diff[np.abs(diff) > 1e-12]
    if diff.size == 0:
        return 0.0
    ranks = rankdata(np.abs(diff))
    pos = ranks[diff > 0].sum()
    neg = ranks[diff < 0].sum()
    return (pos - neg) / (pos + neg)


def _cliffs_delta(diff: np.ndarray) -> float:
    """Cliff's delta for paired data: sign of differences over count."""
    diff = diff[np.abs(diff) > 1e-12]
    if diff.size == 0:
        return 0.0
    pos = np.sum(diff > 0)
    neg = np.sum(diff < 0)
    return (pos - neg) / diff.size


def _bh_q_values(p_values: Iterable[float]) -> np.ndarray:
    p_vals = np.asarray(list(p_values), dtype=float)
    m = p_vals.size
    order = np.argsort(p_vals)
    q_vals = np.empty_like(p_vals)
    prev = 1.0
    for rank, idx in enumerate(order[::-1], start=1):
        p = p_vals[idx]
        q = min(prev, p * m / (m - rank + 1))
        q_vals[idx] = q
        prev = q
    return q_vals


def _bootstrap_ci(diff: np.ndarray, n_boot: int = 5000) -> Tuple[float, float]:
    if diff.size == 0:
        return (np.nan, np.nan)
    idx = RNG.integers(0, diff.size, size=(n_boot, diff.size))
    samples = diff[idx]
    medians = np.median(samples, axis=1)
    return (np.percentile(medians, 2.5), np.percentile(medians, 97.5))


def _friedman_with_kendall(matrix: pd.DataFrame) -> Dict[str, float]:
    values = [matrix[col].to_numpy() for col in matrix.columns]
    stat, p_value = friedmanchisquare(*values)
    n, k = matrix.shape
    ranks = matrix.rank(axis=1, method="average", ascending=False)
    rank_sums = ranks.sum(axis=0).to_numpy()
    q_stat = (12 / (n * k * (k + 1))) * np.sum(rank_sums ** 2) - 3 * n * (k + 1)
    kendall_w = q_stat / (n * (k - 1))
    return {
        "chi_sq": float(stat),
        "p_value": float(p_value),
        "kendall_w": float(kendall_w),
    }


def _pairwise_wilcoxon(matrix: pd.DataFrame) -> pd.DataFrame:
    results = []
    cols = list(matrix.columns)
    for a, b in combinations(cols, 2):
        series_a = matrix[a]
        series_b = matrix[b]
        diff = (series_a - series_b).dropna().to_numpy()
        if diff.size == 0:
            continue
        stat = wilcoxon(series_a, series_b, zero_method="wilcox", correction=False)
        hl = float(np.median(diff))
        r_rb = float(_rank_biserial(diff))
        cliffs = float(_cliffs_delta(diff))
        results.append({
            "model_a": a,
            "model_b": b,
            "hl_estimate": hl,
            "rank_biserial": r_rb,
            "cliffs_delta": cliffs,
            "p_value": float(stat.pvalue),
        })
    df = pd.DataFrame(results)
    if df.empty:
        return df
    df["q_value"] = _bh_q_values(df["p_value"].to_numpy())
    return df


def _cochran_q(matrix: pd.DataFrame) -> Dict[str, float]:
    data = matrix.to_numpy()
    n, k = data.shape
    col_sums = data.sum(axis=0)
    row_sums = data.sum(axis=1)
    total = col_sums.sum()
    numerator = (k - 1) * (k * np.sum(col_sums ** 2) - total ** 2)
    denominator = k * np.sum(row_sums) - np.sum(row_sums ** 2)
    q_stat = numerator / denominator if denominator != 0 else np.nan
    p_value = float(chi2.sf(q_stat, k - 1)) if not np.isnan(q_stat) else np.nan
    return {"q_stat": float(q_stat), "p_value": p_value}


def _pairwise_mcnemar(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = list(matrix.columns)
    for a, b in combinations(cols, 2):
        col_a = matrix[a].astype(int)
        col_b = matrix[b].astype(int)
        mask = ~(col_a.isna() | col_b.isna())
        subset_a = col_a[mask]
        subset_b = col_b[mask]
        b01 = int(((subset_a == 1) & (subset_b == 0)).sum())
        b10 = int(((subset_a == 0) & (subset_b == 1)).sum())
        disc = b01 + b10
        if disc == 0:
            p_val = 1.0
        else:
            p_val = float(binomtest(min(b01, b10), disc, p=0.5, alternative="two-sided").pvalue)
        rate_a = float(subset_a.mean())
        rate_b = float(subset_b.mean())
        risk_diff = rate_a - rate_b
        odds_ratio = np.nan
        if b10 > 0:
            odds_ratio = b01 / b10 if b10 != 0 else np.inf
        rows.append({
            "model_a": a,
            "model_b": b,
            "b01": b01,
            "b10": b10,
            "risk_diff": risk_diff,
            "odds_ratio": odds_ratio,
            "p_value": p_val,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q_value"] = _bh_q_values(df["p_value"].to_numpy())
    return df


def _pairwise_latency(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cols = list(matrix.columns)
    for a, b in combinations(cols, 2):
        diff = (matrix[a] - matrix[b]).dropna().to_numpy()
        if diff.size == 0:
            continue
        stat = wilcoxon(matrix[a], matrix[b], zero_method="wilcox", correction=False)
        hl = float(np.median(diff))
        ci_low, ci_high = _bootstrap_ci(diff)
        rows.append({
            "model_a": a,
            "model_b": b,
            "hl_difference": hl,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "p_value": float(stat.pvalue),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q_value"] = _bh_q_values(df["p_value"].to_numpy())
    return df


def _compute_reliability(rater_df: pd.DataFrame, dimension: str) -> Dict[str, float]:
    subset = rater_df[rater_df["dimension"] == dimension]
    pivot = subset.pivot_table(index=["prompt_id", "model"], columns="rater", values="score")
    pivot = pivot.dropna()
    if pivot.shape[1] < 2:
        return {"quadratic_kappa": np.nan, "icc2k": np.nan, "percent_agreement": np.nan}
    r1 = pivot.iloc[:, 0].to_numpy()
    r2 = pivot.iloc[:, 1].to_numpy()
    # Discretize to the nearest half-point to reflect rubric categories
    r1_disc = (np.round(r1 * 2) / 2).astype(str)
    r2_disc = (np.round(r2 * 2) / 2).astype(str)
    from sklearn.metrics import cohen_kappa_score

    kappa = float(cohen_kappa_score(r1_disc, r2_disc, weights="quadratic"))
    percent_agree = float(np.mean(r1_disc == r2_disc))

    # ICC(2,k)
    data = pivot.to_numpy()
    n, k = data.shape
    row_means = data.mean(axis=1, keepdims=True)
    col_means = data.mean(axis=0, keepdims=True)
    grand_mean = data.mean()
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_error = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    icc = (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)
    return {
        "quadratic_kappa": kappa,
        "icc2k": float(icc),
        "percent_agreement": percent_agree,
    }


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def plot_effect_size_heatmap(pairs: pd.DataFrame, metric: str) -> None:
    labels = [NAME_MAP[m] for m in MODELS_ORDER]
    size = len(labels)
    matrix = np.zeros((size, size))
    matrix[:] = np.nan
    for _, row in pairs.iterrows():
        i = MODELS_ORDER.index(row["model_a"])
        j = MODELS_ORDER.index(row["model_b"])
        matrix[i, j] = row[metric]
        matrix[j, i] = -row[metric]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="RdBu", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(size))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(size))
    ax.set_yticklabels(labels)
    ax.set_title("Rank-biserial Effect Sizes (Pedagogical Score)")
    for i in range(size):
        for j in range(size):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Effect size r")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "effect_size_heatmap.png", dpi=200)
    plt.close(fig)


def plot_latency_forest(latency_pairs: pd.DataFrame) -> None:
    if latency_pairs.empty:
        return
    labels = [
        f"{NAME_MAP[row['model_a']]} - {NAME_MAP[row['model_b']]}" for _, row in latency_pairs.iterrows()
    ]
    effects = latency_pairs["hl_difference"].to_numpy()
    ci_low = latency_pairs["ci_low"].to_numpy()
    ci_high = latency_pairs["ci_high"].to_numpy()
    y_pos = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(effects, y_pos, xerr=[effects - ci_low, ci_high - effects], fmt="o", color="tab:blue")
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Hodges–Lehmann difference in median latency (ms)")
    ax.set_title("Latency Pairwise Contrasts")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "latency_forest.png", dpi=200)
    plt.close(fig)


def plot_inclusion_riskdiff(context_pairs: pd.DataFrame) -> None:
    if context_pairs.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(context_pairs.shape[0])
    diffs = context_pairs["risk_diff"].to_numpy()
    colors = ["tab:green" if val >= 0 else "tab:red" for val in diffs]
    labels = [
        f"{NAME_MAP[row['model_a']]} - {NAME_MAP[row['model_b']]}" for _, row in context_pairs.iterrows()
    ]
    ax.bar(x, diffs, color=colors)
    ax.axhline(0, color="grey", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Risk difference (proportion)")
    ax.set_title("Context Inclusion Paired Differences")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "context_inclusion_diff.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main() -> None:
    prompt_metrics = pd.read_csv(DATA_DIR / "prompt_level_metrics.csv")
    rater_scores = pd.read_csv(DATA_DIR / "rater_scores.csv")

    ped_matrix = _pivot_metric(prompt_metrics, "pedagogical_score")
    ctx_matrix = _pivot_metric(prompt_metrics, "context_score")
    inclusion_matrix = _pivot_metric(prompt_metrics, "context_inclusion")
    latency_matrix = _pivot_metric(prompt_metrics, "latency_median_ms")

    ped_friedman = _friedman_with_kendall(ped_matrix)
    ctx_friedman = _friedman_with_kendall(ctx_matrix)

    ped_pairs = _pairwise_wilcoxon(ped_matrix)
    ctx_pairs = _pairwise_wilcoxon(ctx_matrix)
    inclusion_global = _cochran_q(inclusion_matrix)
    inclusion_pairs = _pairwise_mcnemar(inclusion_matrix)
    latency_pairs = _pairwise_latency(latency_matrix)

    reliability_results = {
        dim: _compute_reliability(rater_scores, dim)
        for dim in ["socratic_quality", "context_fidelity"]
    }

    # Save tabular outputs
    ped_pairs.to_csv(DATA_DIR / "pedagogical_pairwise.csv", index=False)
    ctx_pairs.to_csv(DATA_DIR / "context_pairwise.csv", index=False)
    inclusion_pairs.to_csv(DATA_DIR / "inclusion_pairwise.csv", index=False)
    latency_pairs.to_csv(DATA_DIR / "latency_pairwise.csv", index=False)

    summary = {
        "pedagogical_friedman": ped_friedman,
        "context_friedman": ctx_friedman,
        "inclusion_cochran_q": inclusion_global,
        "reliability": reliability_results,
    }
    import json

    (DATA_DIR / "statistical_analysis_details.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # Figures
    plot_effect_size_heatmap(ped_pairs, "rank_biserial")
    plot_latency_forest(latency_pairs)
    plot_inclusion_riskdiff(inclusion_pairs)

    print("Saved analysis outputs to data/ and figures/ directories.")


if __name__ == "__main__":
    main()
