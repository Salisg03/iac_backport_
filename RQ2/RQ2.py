import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

DATASET_PATH = "all_backports_openstack_with_backport_type.csv"

OUTPUT_CDF = "rq2_time_cdf.pdf"
OUTPUT_BOXPLOTS = "rq2_time_by_project_and_branch_boxplots.pdf"

df = pd.read_csv(DATASET_PATH)
print("RQ2: Timeliness of IaC Backports")
print("\nLoading dataset...")

initial_len = len(df)

df["days_to_backport"] = pd.to_numeric(df["days_to_backport"], errors="coerce")
df = df.dropna(subset=["days_to_backport"])
df = df[df["days_to_backport"] >= 0]

final_len = len(df)

print(f"Initial rows: {initial_len}")
print(f"Valid rows for RQ2: {final_len}")
print(f"Removed rows: {initial_len - final_len}")
print("\n" + "=" * 80)
print("1) Descriptive statistics")
print("=" * 80)

values = df["days_to_backport"]

mean_val = values.mean()
median_val = values.median()
q1_val = values.quantile(0.25)
q3_val = values.quantile(0.75)
max_val = values.max()

under_7 = (values <= 7).mean() * 100
under_30 = (values <= 30).mean() * 100

print(f"Instances       : {final_len:,}")
print(f"Mean            : {mean_val:.2f} days")
print(f"Median          : {median_val:.2f} days")
print(f"Q1              : {q1_val:.2f} days")
print(f"Q3              : {q3_val:.2f} days")
print(f"Max             : {max_val:.2f} days")
print(f"Within 7 days   : {under_7:.2f}%")
print(f"Within 30 days  : {under_30:.2f}%")

sorted_vals = np.sort(values.values)
cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

fig, ax = plt.subplots(figsize=(6.5, 4.2))

ax.plot(sorted_vals, cdf, linewidth=2)
ax.axvline(
    median_val,
    linestyle="--",
    linewidth=1.5,
    label=f"Median: {median_val:.0f} days"
)

ax.set_xlabel("Days to Backport")
ax.set_ylabel("Cumulative Proportion")
ax.set_title("CDF of Backport Time")
ax.set_xlim(left=0, right=max_val * 1.02)
ax.set_ylim(0, 1.01)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_CDF, format="pdf", bbox_inches="tight")
plt.close()

print("\n" + "=" * 80)
print("2) Backport time by top 10 projects")
print("Test: Kruskal-Wallis H-test")
print("=" * 80)

top_projects = df["project"].value_counts().nlargest(10).index.tolist()
df_top_projects = df[df["project"].isin(top_projects)].copy()

project_groups = [
    df_top_projects[df_top_projects["project"] == project]["days_to_backport"].values
    for project in top_projects
]

h_project, p_project = stats.kruskal(*project_groups)

print(f"Top 10 project instances: {len(df_top_projects):,}")
print(f"Kruskal-Wallis H-statistic: {h_project:.2f}")
print(f"p-value: {p_project:.2e}")

print("\nTop 10 project medians:")
project_summary = (
    df_top_projects
    .groupby("project")["days_to_backport"]
    .agg(["count", "median", "mean", "max"])
    .loc[top_projects]
)
print(project_summary.to_string(float_format=lambda x: f"{x:.2f}"))
print("\n" + "=" * 80)
print("3) Backport time by target stable branch")
print("Test: Kruskal-Wallis H-test")
print("=" * 80)

top_branches = df["target_stable_branch"].value_counts().nlargest(8).index.tolist()
df_top_branches = df[df["target_stable_branch"].isin(top_branches)].copy()

branch_groups = [
    df_top_branches[df_top_branches["target_stable_branch"] == branch]["days_to_backport"].values
    for branch in top_branches
]

h_branch, p_branch = stats.kruskal(*branch_groups)

print(f"Top 8 branch instances: {len(df_top_branches):,}")
print(f"Kruskal-Wallis H-statistic: {h_branch:.2f}")
print(f"p-value: {p_branch:.2e}")

print("\nTop 8 branch medians:")
branch_summary = (
    df_top_branches
    .groupby("target_stable_branch")["days_to_backport"]
    .agg(["count", "median", "mean", "max"])
    .loc[top_branches]
)
print(branch_summary.to_string(float_format=lambda x: f"{x:.2f}"))

fig, axes = plt.subplots(2, 1, figsize=(10, 9))

project_plot_data = [
    np.log1p(df_top_projects[df_top_projects["project"] == project]["days_to_backport"].values)
    for project in top_projects
]

axes[0].boxplot(
    project_plot_data,
    tick_labels=top_projects,
    vert=False,
    showfliers=False
)
axes[0].set_title("Backport Time by Top 10 Projects")
axes[0].set_xlabel("log(1 + days to backport)")
axes[0].grid(axis="x", alpha=0.3)

# ---- Branches
branch_plot_data = [
    np.log1p(df_top_branches[df_top_branches["target_stable_branch"] == branch]["days_to_backport"].values)
    for branch in top_branches
]

axes[1].boxplot(
    branch_plot_data,
    tick_labels=top_branches,
    vert=False,
    showfliers=False
)
axes[1].set_title("Backport Time by Target Stable Branch")
axes[1].set_xlabel("log(1 + days to backport)")
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_BOXPLOTS, format="pdf", bbox_inches="tight")
plt.close()

print("Generated files")
print(f"- {OUTPUT_CDF}")
print(f"- {OUTPUT_BOXPLOTS}")

print("\nUse these values in the paper:")
print(f"Instances: {final_len:,}")
print(f"Mean: {mean_val:.2f}")
print(f"Median: {median_val:.2f}")
print(f"Q1: {q1_val:.2f}")
print(f"Q3: {q3_val:.2f}")
print(f"Max: {max_val:.2f}")
print(f"<= 7 days: {under_7:.2f}%")
print(f"<= 30 days: {under_30:.2f}%")
print(f"Project Kruskal H: {h_project:.2f}, p-value: {p_project:.2e}")
print(f"Branch Kruskal H: {h_branch:.2f}, p-value: {p_branch:.2e}")