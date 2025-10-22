"""Utility to print key statistical comparisons for the paper."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "statistical_summary.json"

NAME_MAP = {
    "openai-gpt-4o-mini": "GPT-4o mini",
    "anthropic-claude-3-sonnet-20240229": "Claude 3 Sonnet",
    "google-gemini-2.5-flash": "Gemini 2.5 Flash",
    "cohere-command-r-08-2024": "Command R",
    "meta-llama-Llama-4-Maverick-17B-128E-Instruct": "Llama 4 Maverick",
}

# Pairs to summarize for pedagogical quality contrasts
_KEY_PAIRS = [
    ("google-gemini-2.5-flash", "openai-gpt-4o-mini"),
    ("google-gemini-2.5-flash", "anthropic-claude-3-sonnet-20240229"),
    ("google-gemini-2.5-flash", "cohere-command-r-08-2024"),
    ("google-gemini-2.5-flash", "meta-llama-Llama-4-Maverick-17B-128E-Instruct"),
    ("anthropic-claude-3-sonnet-20240229", "cohere-command-r-08-2024"),
    ("openai-gpt-4o-mini", "cohere-command-r-08-2024"),
]


def _load_stats() -> dict:
    with DATA.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize_pedagogical_pairs(stats: dict) -> str:
    """Format key pairwise Wilcoxon outputs from the summary JSON."""
    results = []
    pairs = stats.get("pedagogical_pairwise", {})
    for model_a, model_b in _KEY_PAIRS:
        key = f"{model_a}__{model_b}"
        flip = False
        if key not in pairs:
            key = f"{model_b}__{model_a}"
            flip = True
        info = pairs[key]
        hl = info["hl_estimate"]
        effect = info["effect_size"]
        q_value = info["q_value"]
        if flip:
            hl = -hl
            effect = -effect
        results.append(
            f"{NAME_MAP[model_a]} vs {NAME_MAP[model_b]}: "
            f"HL={hl:+.3f}, r={effect:+.3f}, q={q_value:.3e}"
        )
    return "\n".join(results)


def main() -> None:
    stats = _load_stats()
    print("Pedagogical contrasts (Wilcoxon + Hodges–Lehmann):")
    print(summarize_pedagogical_pairs(stats))


if __name__ == "__main__":
    main()
