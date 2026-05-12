import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu
import matplotlib.pyplot as plt
DATASET_FILE = "all_backports_openstack_with_backport_type.csv"
OUTPUT_FIGURE = "rq3_clean_vs_modified_boxplots.pdf"
df = pd.read_csv(DATASET_FILE)
df_valid = df[
    (df["master_api_status"] == "ok") &
    (df["bp_api_status"] == "ok") &
    (df["backport_type"].isin(["Clean", "Modified"]))
].copy()

print("RQ3: Review and Validation Effort")
print(f"Valid instances: {len(df_valid)}")


n_clean = (df_valid["backport_type"] == "Clean").sum()
n_modified = (df_valid["backport_type"] == "Modified").sum()

print(f"Clean backports: {n_clean}")
print(f"Modified backports: {n_modified}")

metrics = [
    ("patchsets", "Patchsets"),
    ("human_messages", "Human comments"),
    ("human_reviewers", "Human reviewers"),
    ("review_hours", "Review hours"),
    ("ci_failures", "CI failures"),
]

def fmt_p(p):
    if pd.isna(p):
        return "NA"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"

def to_num(series):
    return pd.to_numeric(series, errors="coerce").dropna()

print("\n" + "=" * 80)
print("1) Master vs Backport")
print("Test: Wilcoxon signed-rank test")
print("=" * 80)

print(
    f"{'Metric':<25}"
    f"{'N':>8}"
    f"{'Master median':>18}"
    f"{'Backport median':>20}"
    f"{'p-value':>15}"
)

print("-" * 86)

for metric, label in metrics:
    master_col = f"master_{metric}"
    bp_col = f"bp_{metric}"

    paired = df_valid[[master_col, bp_col]].copy()
    paired[master_col] = pd.to_numeric(paired[master_col], errors="coerce")
    paired[bp_col] = pd.to_numeric(paired[bp_col], errors="coerce")
    paired = paired.dropna()

    master_values = paired[master_col]
    bp_values = paired[bp_col]

    if len(paired) > 0 and not np.all(master_values == bp_values):
        _, p_value = wilcoxon(master_values, bp_values)
    else:
        p_value = np.nan

    print(
        f"{label:<25}"
        f"{len(paired):>8}"
        f"{master_values.median():>18.2f}"
        f"{bp_values.median():>20.2f}"
        f"{fmt_p(p_value):>15}"
    )
print("2) Clean vs Modified Backports")
print("Test: Mann-Whitney U test")

print(
    f"{'Metric':<25}"
    f"{'N clean':>10}"
    f"{'N modified':>12}"
    f"{'Clean median':>16}"
    f"{'Modified median':>20}"
    f"{'p-value':>15}"
)

for metric, label in metrics:
    bp_col = f"bp_{metric}"

    clean_values = to_num(df_valid[df_valid["backport_type"] == "Clean"][bp_col])
    modified_values = to_num(df_valid[df_valid["backport_type"] == "Modified"][bp_col])

    if len(clean_values) > 0 and len(modified_values) > 0:
        _, p_value = mannwhitneyu(
            clean_values,
            modified_values,
            alternative="two-sided"
        )
    else:
        p_value = np.nan

    print(
        f"{label:<25}"
        f"{len(clean_values):>10}"
        f"{len(modified_values):>12}"
        f"{clean_values.median():>16.2f}"
        f"{modified_values.median():>20.2f}"
        f"{fmt_p(p_value):>15}"
    )
figure_metrics = [
    ("bp_patchsets", "Patchsets"),
    ("bp_human_messages", "Human comments"),
    ("bp_human_reviewers", "Human reviewers"),
    ("bp_review_hours", "Review hours"),
    ("bp_ci_failures", "CI failures"),
]

fig, axes = plt.subplots(1, 5, figsize=(15, 4))

for ax, (col, title) in zip(axes, figure_metrics):
    clean = to_num(df_valid[df_valid["backport_type"] == "Clean"][col])
    modified = to_num(df_valid[df_valid["backport_type"] == "Modified"][col])

    ax.boxplot(
        [np.log1p(clean), np.log1p(modified)],
        tick_labels=["Clean", "Modified"],
        showfliers=False
    )

    ax.set_title(title)
    ax.set_ylabel("log(1 + value)")
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FIGURE, format="pdf", bbox_inches="tight")
plt.close()
print(f"Figure saved: {OUTPUT_FIGURE}")
