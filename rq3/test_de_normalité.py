import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import normaltest, shapiro

# ============================================================
# Configuration
# ============================================================

dataset_file = "all_backports_openstack_enriched_metrics.csv"
output_figure = "normality_diagnostics_qqplots.pdf"
output_table = "normality_diagnostics_summary.csv"

df = pd.read_csv(dataset_file)

df_valid = df[
    (df["master_api_status"] == "ok") &
    (df["bp_api_status"] == "ok")
].copy()

# Variables représentatives pour RQ2 et RQ3
variables = {
    "days_to_backport": "Backport delay",
    "bp_patchsets": "Backport patchsets",
    "bp_human_messages": "Human comments",
    "bp_review_hours": "Review duration"
}

MAX_POINTS_FOR_QQ = 5000
RANDOM_SEED = 42

summary_rows = []

# ============================================================
# Figure: Q-Q plots
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for ax, (col, title) in zip(axes, variables.items()):
    values = pd.to_numeric(df_valid[col], errors="coerce").dropna()

    n = len(values)
    mean = values.mean()
    median = values.median()
    skewness = values.skew()
    max_value = values.max()

    if 3 <= n <= 5000:
        stat, p_value = shapiro(values)
        test_name = "Shapiro-Wilk"
    elif n > 20:
        stat, p_value = normaltest(values)
        test_name = "D'Agostino K2"
    else:
        stat, p_value = np.nan
        test_name = "Skipped"

    summary_rows.append({
        "variable": col,
        "n": n,
        "mean": mean,
        "median": median,
        "max": max_value,
        "skewness": skewness,
        "normality_test": test_name,
        "normality_p_value": p_value
    })

    if n > MAX_POINTS_FOR_QQ:
        values_plot = values.sample(MAX_POINTS_FOR_QQ, random_state=RANDOM_SEED)
    else:
        values_plot = values

    stats.probplot(values_plot, dist="norm", plot=ax)

    ax.set_title(title)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# Sauvegarde en PDF
plt.savefig(output_figure, format="pdf", bbox_inches="tight")
plt.close()

# ============================================================
# Table CSV avec statistiques
# ============================================================

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(output_table, index=False)

print("Saved:")
print(f"- {output_figure}")
print(f"- {output_table}")
print()
print(summary_df.to_string(index=False))