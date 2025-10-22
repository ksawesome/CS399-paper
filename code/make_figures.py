import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
FIG_DIR = Path(__file__).resolve().parent.parent / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
summary = pd.read_csv(DATA_DIR / 'summary_metrics.csv')
errors = pd.read_csv(DATA_DIR / 'error_breakdown.csv')
category_summary = pd.read_csv(DATA_DIR / 'category_summary.csv')
win_counts = pd.read_csv(DATA_DIR / 'category_win_counts.csv')
prompt_metrics = pd.read_csv(DATA_DIR / 'prompt_level_metrics.csv')

# Canonical display names
name_map = {
    'openai-gpt-4o-mini': 'GPT-4o mini',
    'anthropic-claude-3-sonnet-20240229': 'Claude 3 Sonnet',
    'google-gemini-2.5-flash': 'Gemini 2.5 Flash',
    'cohere-command-r-08-2024': 'Command R 08-2024',
    'meta-llama-Llama-4-Maverick-17B-128E-Instruct': 'Llama 4 Maverick 17B'
}
summary['display'] = summary['model'].map(name_map)
errors['display'] = errors['model'].map(name_map)
prompt_metrics['display'] = prompt_metrics['model'].map(name_map)

"""Radar chart data (normalize desirable direction)
Criteria: Pedagogical Quality, Context Fidelity, Safety (1-PII), Cost (lower better),
Latency (lower better), Reliability, Integration, Controllability
"""
crit_spec = [
    ("Pedagogical Quality", "ped_quality_mean", True),
    ("Context Fidelity", "context_fidelity_mean", True),
    ("Safety (1-PII)", "pii_incidents_per_1k", False),  # lower PII rate is better
    ("Cost (1k interactions)", "cost_per_1k_interactions_usd", False),
    ("Latency (median)", "median_latency_ms", False),
    ("Reliability (first-try)", "first_try_success_rate", True),
    ("Integration", "integration_score", True),
    ("Controllability", "controllability_score", True),
]

radar_df = summary.copy()
radar_cols_norm = []
labels = []
for label, col, hi_good in crit_spec:
    labels.append(label)
    arr = radar_df[col].to_numpy()
    if not hi_good:
        # invert: lower is better -> higher is better via max - x
        arr = np.max(arr) - arr
    mn, mx = np.min(arr), np.max(arr)
    norm_col = f"{col}_norm"
    radar_cols_norm.append(norm_col)
    if mx - mn > 1e-9:
        radar_df[norm_col] = (arr - mn) / (mx - mn)
    else:
        radar_df[norm_col] = np.zeros_like(arr)

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

for _, row in radar_df.iterrows():
    values = row[radar_cols_norm].to_numpy()
    values = np.concatenate([values, values[:1]])
    angs = np.concatenate([angles, angles[:1]])
    ax.plot(angs, values, label=row['display'])
    ax.fill(angs, values, alpha=0.08)

ax.set_xticks(angles)
ax.set_xticklabels(labels)
ax.set_yticklabels([])
ax.set_title('Normalized Criteria Radar')
ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
plt.tight_layout()
plt.savefig(FIG_DIR / 'radar_criteria.png', dpi=200)
plt.close()

# Scatter: Pedagogical Quality vs Cost per 1k interactions
plt.figure(figsize=(7, 5))
for _, row in summary.iterrows():
    plt.scatter(row['cost_per_1k_interactions_usd'], row['ped_quality_mean'], s=80)
    plt.text(row['cost_per_1k_interactions_usd']*1.01, row['ped_quality_mean']*1.005, row['display'])
plt.xlabel('Cost per 1k interactions (USD)')
plt.ylabel('Pedagogical Quality (mean 1–5)')
plt.title('Quality vs Cost')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'scatter_quality_cost.png', dpi=200)
plt.close()

# CDF of latency: synthesize request-level samples using median and p95 assuming lognormal-like spread
np.random.seed(42)
plt.figure(figsize=(8, 5))
for _, row in summary.iterrows():
    med = row['median_latency_ms']
    p95 = row['p95_latency_ms']
    # Estimate lognormal sigma from median and p95: p95 = median * exp(z* sigma) with z ~ 1.645
    sigma = np.log(p95/med) / 1.645 if p95>med else 0.2
    mu = np.log(med)
    samples = np.random.lognormal(mean=mu, sigma=sigma, size=5000)
    xs = np.sort(samples)
    ys = np.linspace(0, 1, len(xs))
    plt.plot(xs, ys, label=row['display'])
plt.xlabel('Latency (ms)')
plt.ylabel('CDF')
plt.title('Latency CDF (synthesized)')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'cdf_latency.png', dpi=200)
plt.close()

# Error stacked bar chart
m = errors.merge(summary[['model','display']], on='model', how='left', suffixes=("_err","_sum"))
# prefer existing display from errors; fall back to summary
m['label'] = m.get('display_err', m.get('display_sum'))
if 'display_err' in m.columns and m['display_err'].isna().any():
    m['label'] = m['display_err'].fillna(m.get('display_sum'))
elif 'display_sum' in m.columns and 'label' not in m.columns:
    m['label'] = m['display_sum']

ind = np.arange(len(m))
width = 0.6
plt.figure(figsize=(8, 5))
plt.bar(ind, m['hallucination_pct'], width, label='Hallucination')
plt.bar(ind, m['missing_context_pct'], width, bottom=m['hallucination_pct'], label='Missing Context')
bot2 = m['hallucination_pct'] + m['missing_context_pct']
plt.bar(ind, m['unsafe_tone_pct'], width, bottom=bot2, label='Unsafe/Unhelpful Tone')
bot3 = bot2 + m['unsafe_tone_pct']
plt.bar(ind, m['api_error_pct'], width, bottom=bot3, label='API Error')
plt.xticks(ind, m['label'].astype(str).tolist(), rotation=20)
plt.ylabel('Percent of responses (%)')
plt.title('Error Breakdown by Model')
plt.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / 'errors_stacked.png', dpi=200)
plt.close()

# Category win heatmap
ordered_models = list(name_map.keys())
wins = win_counts.set_index('category')[ordered_models]
wins_display = wins.rename(columns=name_map)
category_labels = []
for cat in wins_display.index:
    label = cat.replace('_', ' ').title()
    label = label.replace('Cs ', 'CS ').replace('Cs-', 'CS-')
    label = label.replace('Ll', 'LL')
    category_labels.append(label)

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(wins_display.values, cmap='Blues', aspect='auto')
ax.set_xticks(np.arange(wins_display.shape[1]))
ax.set_xticklabels(wins_display.columns, rotation=30, ha='right')
ax.set_yticks(np.arange(len(category_labels)))
ax.set_yticklabels(category_labels)

for i in range(wins_display.shape[0]):
    for j in range(wins_display.shape[1]):
        val = int(wins_display.values[i, j])
        ax.text(j, i, f"{val}", ha='center', va='center', color='black', fontsize=8)

ax.set_title('Category Win Counts (Top-1 per Prompt)')
fig.colorbar(im, ax=ax, label='Wins (count)')
plt.tight_layout()
plt.savefig(FIG_DIR / 'category_wins_heatmap.png', dpi=200)
plt.close()

# Prompt-level pedagogical score distribution
model_order = summary.sort_values('ped_quality_mean', ascending=False)['model'].tolist()
label_order = [name_map[m] for m in model_order]
box_data = [
    prompt_metrics[prompt_metrics['model'] == model]['pedagogical_score'].dropna().to_numpy()
    for model in model_order
]

plt.figure(figsize=(8, 5))
bp = plt.boxplot(box_data, patch_artist=True)
palette = plt.get_cmap('viridis')(np.linspace(0.15, 0.85, len(label_order)))
for patch, color in zip(bp['boxes'], palette):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
for whisker in bp['whiskers']:
    whisker.set_color('#444444')
for cap in bp['caps']:
    cap.set_color('#444444')
for median in bp['medians']:
    median.set_color('#222222')

plt.xticks(np.arange(1, len(label_order) + 1), label_order, rotation=15)
plt.ylabel('Pedagogical Score (1–5)')
plt.title('Prompt-Level Pedagogical Score Distribution')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'pedagogical_boxplot.png', dpi=200)
plt.close()

print('Saved figures to', FIG_DIR)
