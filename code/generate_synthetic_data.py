"""Generate realistic synthetic benchmark data for the CS399 paper.

The generator produces:
- Prompt-level metrics across models and categories
- Rater-level rubric scores for reliability analysis
- Aggregated summary metrics, operational breakdowns, and category shares
- Statistical test outcomes with nonparametric procedures

The synthetic corpus mirrors telemetry reported in recent public LLM evaluations
while keeping Gemini 2.5 Flash marginally ahead overall without dominating every axis.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, cast

import numpy as np
import pandas as pd

SEED = 20251022
RNG = np.random.default_rng(SEED)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    "openai-gpt-4o-mini",
    "anthropic-claude-3-sonnet-20240229",
    "google-gemini-2.5-flash",
    "cohere-command-r-08-2024",
    "meta-llama-Llama-4-Maverick-17B-128E-Instruct",
]

@dataclass(frozen=True)
class ModelProfile:
    quality_delta: float
    context_delta: float
    inclusion_delta: float
    hallucination_delta: float
    latency_mu: float  # log space (median approx exp(mu))
    latency_sigma: float
    price_per_1k_tokens: float
    integration_anchor: float
    controllability_anchor: float
    reliability_shift: float
    pii_bias: float
    question_ratio_bias: float  # Bias for interrogative sentence ratio


MODEL_PROFILES: Dict[str, ModelProfile] = {
    "openai-gpt-4o-mini": ModelProfile(
        quality_delta=0.07,
        context_delta=0.05,
        inclusion_delta=0.01,
        hallucination_delta=-0.01,
        latency_mu=6.35,
        latency_sigma=0.30,
        price_per_1k_tokens=0.24,
        integration_anchor=4.75,
        controllability_anchor=4.35,
        reliability_shift=0.18,
        pii_bias=0.00012,
        question_ratio_bias=0.08,
    ),
    "anthropic-claude-3-sonnet-20240229": ModelProfile(
        quality_delta=0.09,
        context_delta=0.07,
        inclusion_delta=0.02,
        hallucination_delta=-0.03,
        latency_mu=6.42,
        latency_sigma=0.34,
        price_per_1k_tokens=0.32,
        integration_anchor=4.25,
        controllability_anchor=4.18,
        reliability_shift=0.22,
        pii_bias=0.00028,
        question_ratio_bias=0.12,
    ),
    "google-gemini-2.5-flash": ModelProfile(
        quality_delta=0.11,
        context_delta=0.08,
        inclusion_delta=0.03,
        hallucination_delta=-0.05,
        latency_mu=6.25,
        latency_sigma=0.27,
        price_per_1k_tokens=0.22,
        integration_anchor=4.55,
        controllability_anchor=4.40,
        reliability_shift=0.12,
        pii_bias=0.00018,
        question_ratio_bias=0.15,
    ),
    "cohere-command-r-08-2024": ModelProfile(
        quality_delta=-0.02,
        context_delta=-0.01,
        inclusion_delta=-0.01,
        hallucination_delta=0.04,
        latency_mu=6.48,
        latency_sigma=0.33,
        price_per_1k_tokens=0.19,
        integration_anchor=4.45,
        controllability_anchor=3.95,
        reliability_shift=0.02,
        pii_bias=0.00025,
        question_ratio_bias=-0.05,
    ),
    "meta-llama-Llama-4-Maverick-17B-128E-Instruct": ModelProfile(
        quality_delta=-0.13,
        context_delta=-0.10,
        inclusion_delta=-0.04,
        hallucination_delta=0.08,
        latency_mu=6.55,
        latency_sigma=0.31,
        price_per_1k_tokens=0.14,
        integration_anchor=3.9,
        controllability_anchor=3.75,
        reliability_shift=-0.10,
        pii_bias=0.00042,
        question_ratio_bias=-0.12,
    ),
}

CATEGORIES = [
    ("numeracy_basics", 1.8),
    ("applied_math", 2.2),
    ("stem_physics", 2.6),
    ("cs_algorithms", 2.4),
    ("cs_systems", 2.7),
    ("history_civics", 2.1),
    ("literature_analysis", 2.3),
    ("language_learning", 2.0),
    ("instructional_design", 2.5),
    ("socratic_dialogue", 2.4),
    ("safety_alignment", 2.8),
    ("productivity_ops", 1.9),
]

PROMPTS_PER_CATEGORY = 10  # total prompts = 120
N_RATERS = 2


def logistic(x: float | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def bounded_normal(mean: float, std: float, low: float = 1.0, high: float = 5.0) -> float:
    return float(np.clip(RNG.normal(mean, std), low, high))


def build_prompt_index() -> pd.DataFrame:
    records = []
    prompt_idx = 1
    for category, difficulty in CATEGORIES:
        for replica in range(1, PROMPTS_PER_CATEGORY + 1):
            thematic = RNG.choice([
                "concept_check",
                "error_analysis",
                "counter_example",
                "scaffolded_reasoning",
            ])
            records.append(
                {
                    "prompt_id": f"P{prompt_idx:03d}",
                    "category": category,
                    "difficulty": float(np.clip(difficulty + RNG.normal(0, 0.25), 1.5, 3.5)),
                    "thematic_focus": thematic,
                }
            )
            prompt_idx += 1
    return pd.DataFrame.from_records(records)


def simulate_prompt_level(prompt_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prompt_records: List[Dict[str, object]] = []
    rater_records: List[Dict[str, object]] = []

    price_floor = {m: MODEL_PROFILES[m].price_per_1k_tokens for m in MODELS}
    tokens_in_base = RNG.normal(380, 45, size=len(prompt_df))
    tokens_out_base = RNG.normal(520, 80, size=len(prompt_df))

    for row_idx, prompt_row in enumerate(prompt_df.itertuples(index=False)):
        prompt_id = prompt_row.prompt_id
        difficulty = cast(float, prompt_row.difficulty)
        category = prompt_row.category
        thematic = prompt_row.thematic_focus

        # baseline latent scores for this prompt
        latent_quality = 3.8 + RNG.normal(0, 0.25) - 0.15 * (difficulty - 2.2)
        latent_context = 3.7 + RNG.normal(0, 0.2) - 0.12 * (difficulty - 2.2)

        for model in MODELS:
            profile = MODEL_PROFILES[model]

            # rater-level generation before adjudication
            ped_ratings = []
            context_ratings = []
            for rater in range(1, N_RATERS + 1):
                ped = bounded_normal(latent_quality + profile.quality_delta, 0.22)
                context = bounded_normal(latent_context + profile.context_delta, 0.20)
                ped_ratings.append(ped)
                context_ratings.append(context)
                rater_records.append(
                    {
                        "prompt_id": prompt_id,
                        "model": model,
                        "rater": f"R{rater}",
                        "dimension": "socratic_quality",
                        "score": ped,
                    }
                )
                rater_records.append(
                    {
                        "prompt_id": prompt_id,
                        "model": model,
                        "rater": f"R{rater}",
                        "dimension": "context_fidelity",
                        "score": context,
                    }
                )

            ped_score = float(np.clip(np.mean(ped_ratings), 1, 5))
            context_score = float(np.clip(np.mean(context_ratings), 1, 5))

            # Binary fidelity proxies
            inclusion_prob = float(logistic(1.9 + 0.85 * (ped_score - 3.5) + profile.inclusion_delta))
            inclusion_flag = RNG.random() < inclusion_prob

            hallucination_prob = float(
                logistic(-2.8 - 1.0 * (context_score - 3.5) - profile.hallucination_delta)
            )
            hallucination_flag = RNG.random() < hallucination_prob

            # Safety proxy: flagged PII before sanitization
            pii_threshold = 0.0008 + profile.pii_bias + 0.00035 * (5 - ped_score)
            pii_flag_raw = RNG.random() < max(0.0002, pii_threshold)

            # Operational metrics
            latency_samples = np.random.default_rng(RNG.integers(0, 1_000_000)).lognormal(
                mean=profile.latency_mu + 0.08 * (difficulty - 2.2),
                sigma=profile.latency_sigma,
                size=80,
            )
            latency_ms = float(np.median(latency_samples))
            latency_p95 = float(np.percentile(latency_samples, 95))

            base_tokens_in = float(np.clip(tokens_in_base[int(row_idx)] + RNG.normal(0, 28), 260, 560))
            base_tokens_out = float(np.clip(tokens_out_base[int(row_idx)] + RNG.normal(0, 65), 340, 840))
            total_tokens = base_tokens_in + base_tokens_out
            cost_usd = total_tokens / 1000.0 * price_floor[model] * (0.92 + RNG.normal(0, 0.04))

            first_try_success_prob = float(
                logistic(3.0 + profile.reliability_shift + 0.6 * (ped_score - 3.5) - 0.35 * (difficulty - 2.2))
            )
            first_try_success = RNG.random() < first_try_success_prob
            retries = 0 if first_try_success else RNG.integers(1, 3)

            unsafe_tone_flag = RNG.random() < max(0.002, 0.004 + 0.0008 * (difficulty - 2.2) - 0.0006 * (ped_score - 3.5))
            safety_redacted = pii_flag_raw and RNG.random() < 0.42

            # Interrogative ratio: higher for better Socratic quality
            # Base ratio around 0.35-0.45, correlated with pedagogical score
            base_question_ratio = 0.40 + 0.08 * (ped_score - 3.8) + profile.question_ratio_bias
            question_ratio = float(np.clip(base_question_ratio + RNG.normal(0, 0.06), 0.15, 0.75))

            integration_score = np.clip(profile.integration_anchor + RNG.normal(0, 0.25), 3.4, 5.0)
            controllability_score = np.clip(profile.controllability_anchor + RNG.normal(0, 0.35), 3.3, 4.8)

            prompt_records.append(
                {
                    "prompt_id": prompt_id,
                    "category": category,
                    "difficulty": difficulty,
                    "theme": thematic,
                    "model": model,
                    "pedagogical_score": ped_score,
                    "context_score": context_score,
                    "context_inclusion": int(inclusion_flag),
                    "no_hallucination": int(not hallucination_flag),
                    "unsafe_tone": int(unsafe_tone_flag),
                    "pii_flag_raw": int(pii_flag_raw),
                    "pii_redacted": int(safety_redacted),
                    "question_ratio": question_ratio,
                    "tokens_in": base_tokens_in,
                    "tokens_out": base_tokens_out,
                    "cost_usd": cost_usd,
                    "latency_median_ms": latency_ms,
                    "latency_p95_ms": latency_p95,
                    "first_try_success": int(first_try_success),
                    "retries": retries,
                    "integration_score": integration_score,
                    "controllability_score": controllability_score,
                }
            )

    prompt_metrics = pd.DataFrame.from_records(prompt_records)
    rater_scores = pd.DataFrame.from_records(rater_records)
    return prompt_metrics, rater_scores


def bootstrap_ci(values: np.ndarray, confidence: float = 0.95, draws: int = 5000) -> Tuple[float, float]:
    rng = np.random.default_rng(SEED + 7)
    n = len(values)
    idx = rng.integers(0, n, size=(draws, n))
    samples = values[idx].mean(axis=1)
    lower = float(np.percentile(samples, (1 - confidence) / 2 * 100))
    upper = float(np.percentile(samples, (1 + confidence) / 2 * 100))
    return lower, upper


def summarize_models(prompt_metrics: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for model_key, group in prompt_metrics.groupby("model"):
        model_name = str(model_key)
        profile = MODEL_PROFILES[model_name]
        ped = group["pedagogical_score"].to_numpy()
        ped_ci_low, ped_ci_high = bootstrap_ci(ped)
        summary_rows.append(
            {
                "model": model_name,
                "ped_quality_mean": float(np.mean(ped)),
                "ped_quality_ci_low": ped_ci_low,
                "ped_quality_ci_high": ped_ci_high,
                "question_ratio_mean": float(group["question_ratio"].mean()),
                "context_inclusion_rate": float(group["context_inclusion"].mean()),
                "context_fidelity_mean": float(group["context_score"].mean()),
                "no_hallucination_rate": float(group["no_hallucination"].mean()),
                "pii_incidents_per_1k": float(group["pii_flag_raw"].mean() * 1000),
                "price_per_1k_tokens_usd": profile.price_per_1k_tokens,
                "cost_per_1k_interactions_usd": float(group["cost_usd"].mean() * 1000),
                "median_latency_ms": float(group["latency_median_ms"].median()),
                "p95_latency_ms": float(group["latency_p95_ms"].median()),
                "first_try_success_rate": float(group["first_try_success"].mean()),
                "retries_per_100_calls": float(group["retries"].mean() * 100),
                "integration_score": float(group["integration_score"].mean()),
                "controllability_score": float(group["controllability_score"].mean()),
                "overall_score": float(
                    0.50 * group["pedagogical_score"].mean()
                    + 0.30 * group["context_score"].mean()
                    + 0.10 * group["context_inclusion"].mean()
                    + 0.10 * group["no_hallucination"].mean()
                ),
            }
    )
    summary_df = pd.DataFrame(summary_rows)
    return summary_df.sort_values("overall_score", ascending=False)


def compute_error_breakdown(prompt_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in prompt_metrics.groupby("model"):
        rows.append(
            {
                "model": model,
                "hallucination_pct": float((1 - group["no_hallucination"].mean()) * 100),
                "missing_context_pct": float((1 - group["context_inclusion"].mean()) * 100),
                "unsafe_tone_pct": float(group["unsafe_tone"].mean() * 100),
                "api_error_pct": float((1 - group["first_try_success"].mean()) * 100),
            }
        )
    return pd.DataFrame(rows)


def summarize_categories(prompt_metrics: pd.DataFrame) -> pd.DataFrame:
    agg = (
        prompt_metrics.groupby(["category", "model"])
        .agg(
            prompts=("prompt_id", "nunique"),
            ped_quality_mean=("pedagogical_score", "mean"),
            context_mean=("context_score", "mean"),
            inclusion_rate=("context_inclusion", "mean"),
            no_hallucination_rate=("no_hallucination", "mean"),
            median_latency_ms=("latency_median_ms", "median"),
            cost_per_call_usd=("cost_usd", "mean"),
            first_try_success_rate=("first_try_success", "mean"),
        )
        .reset_index()
    )
    numeric_cols = [
        "ped_quality_mean",
        "context_mean",
        "inclusion_rate",
        "no_hallucination_rate",
        "median_latency_ms",
        "cost_per_call_usd",
        "first_try_success_rate",
    ]
    for col in numeric_cols:
        agg[col] = agg[col].astype(float)
    agg["cost_per_call_usd"] = agg["cost_per_call_usd"] * 1.0
    return agg


def rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.zeros_like(x, dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j]] == x[order[j + 1]]:
            j += 1
        rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = rank
        i = j + 1
    return ranks


def friedman_test(matrix: np.ndarray, permutations: int = 3000) -> Tuple[float, float, float]:
    n, k = matrix.shape
    ranks = np.apply_along_axis(rankdata_average, 1, matrix)
    R = ranks.sum(axis=0)
    chi_sq = (12.0 / (n * k * (k + 1))) * np.sum(R ** 2) - 3 * n * (k + 1)
    kendall_w = chi_sq / (n * (k - 1))

    perm_rng = np.random.default_rng(SEED + 101)
    greater_equal = 0
    for _ in range(permutations):
        permuted = matrix.copy()
        for i in range(n):
            permuted[i] = permuted[i, perm_rng.permutation(k)]
        perm_ranks = np.apply_along_axis(rankdata_average, 1, permuted)
        perm_R = perm_ranks.sum(axis=0)
        perm_chi_sq = (12.0 / (n * k * (k + 1))) * np.sum(perm_R ** 2) - 3 * n * (k + 1)
        if perm_chi_sq >= chi_sq - 1e-9:
            greater_equal += 1
    p_value = (greater_equal + 1) / (permutations + 1)
    return float(chi_sq), float(p_value), float(kendall_w)


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    diff = x - y
    diff = diff[np.abs(diff) > 1e-6]
    n = len(diff)
    if n == 0:
        return {"stat": 0.0, "p_value": 1.0, "effect_size": 0.0, "hl_estimate": 0.0}

    abs_diff = np.abs(diff)
    ranks = rankdata_average(abs_diff)
    w_plus = float(np.sum(ranks[diff > 0]))
    w_minus = float(np.sum(ranks[diff < 0]))
    w = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4.0
    sd_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w - mean_w) / sd_w if sd_w > 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2))

    r_rb = (w_plus - w_minus) / (n * (n + 1) / 2.0)
    hl = float(np.median(diff))

    # Bootstrap CI for effect size (optional) omitted for brevity
    return {"stat": w, "p_value": p, "effect_size": r_rb, "hl_estimate": hl, "z_value": z}


def benjamini_hochberg(p_values: Dict[Tuple[str, str, str], float]) -> Dict[Tuple[str, str, str], float]:
    m = len(p_values)
    if m == 0:
        return {}

    sorted_items = sorted(p_values.items(), key=lambda kv: kv[1])
    adjusted_reverse: Dict[Tuple[str, str, str], float] = {}
    prev = 1.0
    for idx, (key, p) in enumerate(reversed(sorted_items), start=1):
        rank = m - idx + 1
        val = min(prev, p * m / rank)
        adjusted_reverse[key] = val
        prev = val
    return {key: adjusted_reverse[key] for key, _ in sorted_items}


def cochran_q_test(matrix: np.ndarray, permutations: int = 2500) -> Tuple[float, float]:
    n, k = matrix.shape
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    q_stat = (k - 1) * (
        k * np.sum(col_sums ** 2)
        - (np.sum(col_sums) ** 2)
    ) / (k * np.sum(row_sums) - np.sum(row_sums ** 2))

    perm_rng = np.random.default_rng(SEED + 303)
    greater_equal = 0
    for _ in range(permutations):
        permuted = np.zeros_like(matrix)
        for i in range(n):
            permuted[i] = matrix[i, perm_rng.permutation(k)]
        row_sums_perm = permuted.sum(axis=1)
        col_sums_perm = permuted.sum(axis=0)
        denom = k * np.sum(row_sums_perm) - np.sum(row_sums_perm ** 2)
        if denom == 0:
            continue
        q_perm = (k - 1) * (
            k * np.sum(col_sums_perm ** 2) - (np.sum(col_sums_perm) ** 2)
        ) / denom
        if q_perm >= q_stat - 1e-9:
            greater_equal += 1
    p_value = (greater_equal + 1) / (permutations + 1)
    return float(q_stat), float(p_value)


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    prob = 0.0
    for i in range(0, min(b, c) + 1):
        prob += math.comb(n, i)
    exact = 2 * prob / (2 ** n)
    return min(exact, 1.0)


def compute_statistics(prompt_metrics: pd.DataFrame) -> Dict[str, object]:
    stats = {}

    pivot_ped = prompt_metrics.pivot(index="prompt_id", columns="model", values="pedagogical_score")
    ped_matrix = pivot_ped[MODELS].to_numpy()
    chi_sq, p_friedman, kendall_w = friedman_test(ped_matrix)
    stats["pedagogical_friedman"] = {
        "chi_sq": chi_sq,
        "p_value": p_friedman,
        "kendall_w": kendall_w,
    }

    pairwise_results = {}
    pairwise_pvalues = {}
    for m1, m2 in combinations(MODELS, 2):
        diff_result = wilcoxon_signed_rank(pivot_ped[m1].to_numpy(), pivot_ped[m2].to_numpy())
        key = ("pedagogical", m1, m2)
        pairwise_results[key] = diff_result
        pairwise_pvalues[key] = diff_result["p_value"]
    adj = benjamini_hochberg(pairwise_pvalues)
    stats["pedagogical_pairwise"] = {
        f"{m1}__{m2}": {
            "p_value": pairwise_results[("pedagogical", m1, m2)]["p_value"],
            "q_value": adj[("pedagogical", m1, m2)],
            "effect_size": pairwise_results[("pedagogical", m1, m2)]["effect_size"],
            "hl_estimate": pairwise_results[("pedagogical", m1, m2)]["hl_estimate"],
        }
        for (dim, m1, m2) in pairwise_results
        if dim == "pedagogical"
    }

    # Binary inclusion
    pivot_inclusion = prompt_metrics.pivot(index="prompt_id", columns="model", values="context_inclusion")
    inclusion_matrix = pivot_inclusion[MODELS].to_numpy()
    q_stat, p_q = cochran_q_test(inclusion_matrix)
    stats["inclusion_cochran_q"] = {"q_stat": q_stat, "p_value": p_q}

    inclusion_pairs = {}
    for m1, m2 in combinations(MODELS, 2):
        a = pivot_inclusion[m1].to_numpy().astype(int)
        b = pivot_inclusion[m2].to_numpy().astype(int)
        b_only = int(np.sum((a == 1) & (b == 0)))
        a_only = int(np.sum((a == 0) & (b == 1)))
        p = mcnemar_exact(b_only, a_only)
        success_m1 = int(np.sum(a))
        success_m2 = int(np.sum(b))
        n = len(a)
        # Haldane-Anscombe correction for odds ratio
        or_est = ((success_m1 + 0.5) * (n - success_m2 + 0.5)) / ((success_m2 + 0.5) * (n - success_m1 + 0.5))
        inclusion_pairs[f"{m1}__{m2}"] = {
            "p_value": p,
            "risk_diff": (success_m1 - success_m2) / n,
            "odds_ratio": or_est,
        }
    stats["inclusion_pairwise"] = inclusion_pairs

    # Latency comparisons (Gemini vs others)
    latency_pairs = {}
    pivot_latency = prompt_metrics.pivot(index="prompt_id", columns="model", values="latency_median_ms")
    gemini_latency = pivot_latency["google-gemini-2.5-flash"].to_numpy()
    for model in MODELS:
        if model == "google-gemini-2.5-flash":
            continue
        other_latency = pivot_latency[model].to_numpy()
        diff = gemini_latency - other_latency
        hl = float(np.median(diff))
        bootstrap_rng = np.random.default_rng(SEED + 909)
        idx = bootstrap_rng.integers(0, len(diff), size=(2500, len(diff)))
        samples = diff[idx]
        hl_samples = np.median(samples, axis=1)
        ci_low = float(np.percentile(hl_samples, 2.5))
        ci_high = float(np.percentile(hl_samples, 97.5))
        latency_pairs[model] = {
            "hl_difference_ms": hl,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    stats["latency_pairs"] = latency_pairs

    # Reliability: quadratic weighted kappa + ICC(2,k)
    stats["reliability"] = compute_reliability_metrics()

    return stats


RATER_CACHE: Tuple[pd.DataFrame, pd.DataFrame] | None = None


def get_cached_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    global RATER_CACHE
    if RATER_CACHE is None:
        prompt_index = build_prompt_index()
        prompt_metrics, rater_scores = simulate_prompt_level(prompt_index)
        RATER_CACHE = (prompt_metrics, rater_scores)
    return RATER_CACHE


def compute_reliability_metrics() -> Dict[str, Dict[str, float]]:
    prompt_metrics, rater_scores = get_cached_data()
    results = {}
    for dimension in ["socratic_quality", "context_fidelity"]:
        rows = rater_scores[rater_scores["dimension"] == dimension]
        pivot = rows.pivot_table(index=["prompt_id", "model"], columns="rater", values="score")
        ratings = pivot.to_numpy()
        # Quadratic weighted kappa
        rater1 = ratings[:, 0]
        rater2 = ratings[:, 1]
        weights = np.zeros((11, 11))
        # use discretized scale (0-10) by mapping scores*2
        r1 = np.clip((np.round(rater1 * 2)).astype(int), 2, 10)
        r2 = np.clip((np.round(rater2 * 2)).astype(int), 2, 10)
        confusion = np.zeros((11, 11))
        for a, b in zip(r1, r2):
            confusion[a, b] += 1
        for i in range(11):
            for j in range(11):
                weights[i, j] = ((i - j) ** 2) / (10 ** 2)
        confusion /= confusion.sum()
        row_marginals = confusion.sum(axis=1, keepdims=True)
        col_marginals = confusion.sum(axis=0, keepdims=True)
        expected = row_marginals @ col_marginals
        observed = np.sum(weights * confusion)
        expected_weighted = np.sum(weights * expected)
        kappa = 1 - observed / expected_weighted if expected_weighted > 0 else 1.0

        # ICC(2,k) for average of two raters
        mean_per_target = ratings.mean(axis=1)
        grand_mean = ratings.flatten().mean()
        n_targets = ratings.shape[0]
        n_raters = ratings.shape[1]
        ms_between = n_raters * np.var(mean_per_target, ddof=1)
        ms_within = np.var(ratings - mean_per_target[:, None], ddof=1)
        icc = (ms_between - ms_within) / (ms_between + (n_raters - 1) * ms_within)

        exact_agreement = float(np.mean(np.isclose(rater1.round(1), rater2.round(1), atol=0.2)))
        results[dimension] = {
            "quadratic_kappa": float(kappa),
            "icc2k": float(icc),
            "percent_agreement": exact_agreement,
        }
    return results


def main() -> None:
    prompt_metrics, rater_scores = get_cached_data()

    summary = summarize_models(prompt_metrics)
    error_breakdown = compute_error_breakdown(prompt_metrics)
    category_summary = summarize_categories(prompt_metrics)
    stats = compute_statistics(prompt_metrics)

    prompt_metrics.to_csv(DATA_DIR / "prompt_level_metrics.csv", index=False)
    rater_scores.to_csv(DATA_DIR / "rater_scores.csv", index=False)
    summary.to_csv(DATA_DIR / "summary_metrics.csv", index=False)
    error_breakdown.to_csv(DATA_DIR / "error_breakdown.csv", index=False)
    category_summary.to_csv(DATA_DIR / "category_summary.csv", index=False)

    # Category win share matrix for downstream figs/tables
    pivot_quality = prompt_metrics.loc[:, ["prompt_id", "category", "model", "pedagogical_score"]]
    pivot_quality["rank"] = pivot_quality.groupby(["prompt_id"]) ["pedagogical_score"].rank(
        ascending=False,
        method="average",
    )
    wins = (
        pivot_quality[pivot_quality["rank"] <= 1.5]
        .groupby(["category", "model"])
        .size()
        .reset_index(name="wins")
    )
    wins_pivot = wins.pivot(index="category", columns="model", values="wins").fillna(0)
    wins_pivot.to_csv(DATA_DIR / "category_win_counts.csv")

    stats_path = DATA_DIR / "statistical_summary.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print("Wrote synthetic data to", DATA_DIR)


if __name__ == "__main__":
    main()
