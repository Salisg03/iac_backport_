import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
DATASET_FILE = "all_backports_openstack_with_backport_type.csv"
OUTPUT_FIGURE = "rq4_patch_differences.pdf"

def pct(x, total):
    return (x / total * 100) if total > 0 else 0.0


def pformat(p):
    if pd.isna(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.4f}"


def describe_series(series, name):
    s = pd.to_numeric(series, errors="coerce").dropna()
    print(f"\n{name}")
    print("-" * 60)
    print(f"N      : {len(s)}")
    if len(s) == 0:
        return
    print(f"Mean   : {s.mean():.2f}")
    print(f"Median : {s.median():.2f}")
    print(f"Q1     : {s.quantile(0.25):.2f}")
    print(f"Q3     : {s.quantile(0.75):.2f}")
    print(f"Min    : {s.min():.2f}")
    print(f"Max    : {s.max():.2f}")


def wilcoxon_test(a, b, label):
    paired = pd.DataFrame({"a": a, "b": b}).dropna()

    print(f"\n[Wilcoxon] {label}")
    print("-" * 60)
    print(f"Paired instances: {len(paired)}")

    if len(paired) == 0:
        print("Test skipped: no paired data")
        return np.nan, np.nan

    diffs = paired["a"] - paired["b"]
    if (diffs == 0).all():
        print("Test skipped: all paired differences are zero")
        return np.nan, np.nan

    stat, p = wilcoxon(paired["a"], paired["b"])
    print(f"W-statistic: {stat:.2f}")
    print(f"p-value    : {pformat(p)}")
    return stat, p

df = pd.read_csv(DATASET_FILE)

df_valid = df[df["backport_type"].isin(["Clean", "Modified"])].copy()
print("RQ4: Patch Differences and Adaptation Patterns")
print(f"Valid instances: {len(df_valid)}")
numeric_cols = [
    "file_overlap_ratio",
    "diff_size_ratio",
    "master_lines_added",
    "master_lines_removed",
    "stable_lines_added",
    "stable_lines_removed",
]

for col in numeric_cols:
    df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")

same_files = (
    df_valid["same_files_modified"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df_valid["same_files_bool"] = same_files.isin(["true", "1", "yes"])

df_valid["master_total_lines"] = (
    df_valid["master_lines_added"].fillna(0) +
    df_valid["master_lines_removed"].fillna(0)
)

df_valid["stable_total_lines"] = (
    df_valid["stable_lines_added"].fillna(0) +
    df_valid["stable_lines_removed"].fillna(0)
)

df_valid["added_line_delta"] = (
    df_valid["stable_lines_added"] - df_valid["master_lines_added"]
)

df_valid["removed_line_delta"] = (
    df_valid["stable_lines_removed"] - df_valid["master_lines_removed"]
)

df_valid["total_line_delta"] = (
    df_valid["stable_total_lines"] - df_valid["master_total_lines"]
)

# Modified reasons
df_valid["file_difference"] = ~df_valid["same_files_bool"]
df_valid["size_difference"] = df_valid["diff_size_ratio"] != 1

df_valid["modified_reason"] = "Clean"

df_valid.loc[
    (df_valid["file_difference"]) & (~df_valid["size_difference"]),
    "modified_reason"
] = "Different files only"

df_valid.loc[
    (~df_valid["file_difference"]) & (df_valid["size_difference"]),
    "modified_reason"
] = "Diff size only"

df_valid.loc[
    (df_valid["file_difference"]) & (df_valid["size_difference"]),
    "modified_reason"
] = "Different files and diff size"

df_clean = df_valid[df_valid["backport_type"] == "Clean"].copy()
df_modified = df_valid[df_valid["backport_type"] == "Modified"].copy()


print("1) Clean vs Modified")

n_total = len(df_valid)
n_clean = len(df_clean)
n_modified = len(df_modified)

print(f"Clean backports    : {n_clean} ({pct(n_clean, n_total):.2f}%)")
print(f"Modified backports : {n_modified} ({pct(n_modified, n_total):.2f}%)")

print("2) Why are backports Modified?")


reason_counts = (
    df_modified["modified_reason"]
    .value_counts()
    .reindex([
        "Different files only",
        "Diff size only",
        "Different files and diff size"
    ])
    .fillna(0)
    .astype(int)
)

for reason, count in reason_counts.items():
    print(f"{reason:<35}: {count:>5} ({pct(count, n_modified):6.2f}%)")



print("3) File overlap ratio (Modified backports)")
describe_series(df_modified["file_overlap_ratio"], "file_overlap_ratio")
print("4) Diff size ratio")
describe_series(df_modified["diff_size_ratio"], "Modified backports: diff_size_ratio")
smaller = (df_modified["diff_size_ratio"] < 1).sum()
equal = (df_modified["diff_size_ratio"] == 1).sum()
larger = (df_modified["diff_size_ratio"] > 1).sum()

print("\nDirection of size change among Modified")
print(f"Stable diff smaller than master : {smaller} ({pct(smaller, n_modified):.2f}%)")
print(f"Stable diff same size as master : {equal} ({pct(equal, n_modified):.2f}%)")
print(f"Stable diff larger than master  : {larger} ({pct(larger, n_modified):.2f}%)")

print("5) Line-level differences (Modified backports)")

describe_series(df_modified["master_total_lines"], "master_total_lines")
describe_series(df_modified["stable_total_lines"], "stable_total_lines")
describe_series(df_modified["total_line_delta"], "stable_total_lines - master_total_lines")

# Paired test: master vs stable among Modified
wilcoxon_test(
    df_modified["master_total_lines"],
    df_modified["stable_total_lines"],
    "Modified backports: master_total_lines vs stable_total_lines"
)

print("6) Top 10 projects by percentage of Modified backports")

project_summary = (
    df_valid.groupby("project")
    .agg(
        total=("instance_id", "count"),
        modified=("backport_type", lambda x: (x == "Modified").sum())
    )
)

project_summary["modified_pct"] = (
    project_summary["modified"] / project_summary["total"] * 100
)

project_summary = (
    project_summary[project_summary["total"] >= 20]
    .sort_values(["modified_pct", "modified"], ascending=[False, False])
    .head(10)
)

print(project_summary.to_string(float_format=lambda x: f"{x:.2f}"))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
overlap_vals = df_modified["file_overlap_ratio"].dropna()

axes[0].hist(overlap_vals, bins=20, edgecolor="white", linewidth=0.3)
axes[0].set_title("File overlap ratio")
axes[0].set_xlabel("File overlap ratio")
axes[0].set_ylabel("Count")
axes[0].grid(axis="y", alpha=0.3)

labels_b = ["Smaller\nthan master", "Same size", "Larger\nthan master"]
values_b = [
    pct(smaller, n_modified),
    pct(equal, n_modified),
    pct(larger, n_modified),
]

bars = axes[1].bar(labels_b, values_b, edgecolor="white", linewidth=0.3)
axes[1].set_title("Direction of size change")
axes[1].set_ylabel("% of Modified backports")
axes[1].set_ylim(0, max(values_b) * 1.20 if max(values_b) > 0 else 1)
axes[1].grid(axis="y", alpha=0.3)

for bar, val in zip(bars, values_b):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.1f}%",
        ha="center",
        fontsize=9
    )

scatter_df = df_modified[["master_total_lines", "stable_total_lines"]].dropna()

x_scatter = np.log1p(scatter_df["master_total_lines"])
y_scatter = np.log1p(scatter_df["stable_total_lines"])

axes[2].scatter(
    x_scatter,
    y_scatter,
    alpha=0.3,
    s=10,
    rasterized=True
)

if len(scatter_df) > 0:
    lim_max = max(x_scatter.max(), y_scatter.max()) + 0.2
else:
    lim_max = 1

axes[2].plot(
    [0, lim_max],
    [0, lim_max],
    linestyle="--",
    linewidth=1,
    label="y = x"
)

axes[2].set_title("Patch size: master vs stable")
axes[2].set_xlabel("log(1 + master total lines)")
axes[2].set_ylabel("log(1 + stable total lines)")
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_FIGURE, format="pdf", bbox_inches="tight", dpi=300)
plt.close()

print("Generated files")
print(f"- {OUTPUT_FIGURE}")

print("PAPER SUMMARY")
print(f"Valid instances: {n_total}")
print(f"Clean backports: {n_clean} ({pct(n_clean, n_total):.2f}%)")
print(f"Modified backports: {n_modified} ({pct(n_modified, n_total):.2f}%)")

print("\nModified reasons:")
for reason, count in reason_counts.items():
    print(f"- {reason}: {count} ({pct(count, n_modified):.2f}%)")

print("\nDirection of size change among Modified:")
print(f"- Smaller than master: {smaller} ({pct(smaller, n_modified):.2f}%)")
print(f"- Same size as master: {equal} ({pct(equal, n_modified):.2f}%)")
print(f"- Larger than master: {larger} ({pct(larger, n_modified):.2f}%)")

if len(overlap_vals) > 0:
    print("\nFile overlap ratio among Modified:")
    print(f"- Median: {overlap_vals.median():.2f}")
    print(f"- Q1: {overlap_vals.quantile(0.25):.2f}")
    print(f"- Q3: {overlap_vals.quantile(0.75):.2f}")

paired_df = df_modified[["master_total_lines", "stable_total_lines"]].dropna()
if len(paired_df) > 0:
    print("\nPatch size among Modified:")
    print(f"- Master median total lines: {paired_df['master_total_lines'].median():.2f}")
    print(f"- Stable median total lines: {paired_df['stable_total_lines'].median():.2f}")