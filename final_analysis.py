import os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import linregress

DATA_FILE = "datafile.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()
df["State/UT"] = df["State/UT"].astype(str).str.strip()
df = df[~df["State/UT"].str.lower().isin(["total", "grand total", "overall"])].copy()
df.drop(columns=["Sl. No."], errors="ignore", inplace=True)
year_columns = [c for c in df.columns if "Road Length Completed" in c]
short_years = ["2021-22", "2022-23", "2023-24", "2024-25"]
for col in year_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.to_csv(f"{RESULTS_DIR}/cleaned_data.csv", index=False)
X_orig = df[year_columns].copy()
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X_orig), columns=year_columns, index=df.index)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

k_values = list(range(2, 8))
eval_rows, km_labels_map, hc_labels_map = [], {}, {}
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    lbl_km = km.fit_predict(X_scaled)
    km_labels_map[k] = lbl_km
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    lbl_hc = hc.fit_predict(X_scaled)
    hc_labels_map[k] = lbl_hc
    null_sils = [silhouette_score(np.random.default_rng(100 + i).normal(0.0, 1.0, size=X_scaled.shape), KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(np.random.default_rng(100 + i).normal(0.0, 1.0, size=X_scaled.shape))) for i in range(20)]
    eval_rows.append({"k": k, "kmeans_silhouette": float(silhouette_score(X_scaled, lbl_km)), "hierarchical_silhouette": float(silhouette_score(X_scaled, lbl_hc)), "null_baseline_silhouette": float(np.mean(null_sils))})
eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(f"{RESULTS_DIR}/clustering_evaluation.csv", index=False)

complete_mask = df[year_columns].notna().all(axis=1)
X_log = StandardScaler().fit_transform(np.log1p(X_imp))
lbl_log_k2 = KMeans(n_clusters=2, random_state=42, n_init=20).fit_predict(X_log)
lbl_log_k3 = KMeans(n_clusters=3, random_state=42, n_init=20).fit_predict(X_log)

time_idx = np.arange(len(year_columns))
trend_rows = []
for _, row in df.iterrows():
    vals = pd.to_numeric(row[year_columns], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(vals)
    if valid.sum() >= 2:
        slope, intercept, r_val, p_val, _ = linregress(time_idx[valid], vals[valid])
        trend_rows.append({"State/UT": row["State/UT"], "Slope_km_per_yr": round(float(slope), 2)})
    else:
        trend_rows.append({"State/UT": row["State/UT"], "Slope_km_per_yr": np.nan})
pd.DataFrame(trend_rows).to_csv(f"{RESULTS_DIR}/road_completion_trends.csv", index=False)
year_summary = pd.DataFrame([{"Financial_Year": short_years[i], "Total_Road_Length": round(float(df[col].sum(min_count=1)), 2), "Missing_States": int(df[col].isna().sum())} for i, col in enumerate(year_columns)])

plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False})

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))
bars = axes[0].bar(short_years, year_summary["Total_Road_Length"] / 1000.0, color="#1f3a60", width=0.52)
axes[0].set_ylabel("Total completed (thousand km)")
axes[0].set_title("(a) Aggregate recorded road completion")
for b in bars: axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 1.0, f"{b.get_height():.1f}k", ha="center", fontsize=8.5)
bars2 = axes[1].bar(short_years, year_summary["Missing_States"], color="#c85a32", width=0.52)
axes[1].set_ylabel("Missing State/UT entries")
axes[1].set_title("(b) Missing values by reporting year")
for b in bars2: axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.12, f"{int(b.get_height())}", ha="center", fontsize=8.5)
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig1_year_audit.png", dpi=220); plt.close()

fig, ax = plt.subplots(figsize=(5.8, 3.4))
ax.plot(eval_df["k"], eval_df["kmeans_silhouette"], marker="o", color="#1f3a60", label="K-Means (raw standardised)")
ax.plot(eval_df["k"], eval_df["hierarchical_silhouette"], marker="s", color="#c85a32", label="Hierarchical (Ward)")
ax.plot(eval_df["k"], eval_df["null_baseline_silhouette"], linestyle="--", marker="^", color="#555555", label="Gaussian null baseline")
ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("Mean Silhouette Score")
ax.set_title("Cluster separation across k vs. Gaussian null baseline")
ax.legend(frameon=False, loc="upper right", fontsize=8.5)
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig2_silhouette_comparison.png", dpi=220); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.9))
c2 = pd.Series(km_labels_map[2]).value_counts().sort_index()
axes[0].bar([f"C{i}" for i in c2.index], c2.values, color=["#1f3a60", "#c85a32"])
axes[0].set_title("(a) Raw K-Means (k=2)"); axes[0].set_ylabel("States / UTs")
for i, v in enumerate(c2.values): axes[0].text(i, v + 0.8, str(v), ha="center", fontsize=8.5)
c3 = pd.Series(km_labels_map[3]).value_counts().sort_index()
axes[1].bar([f"C{i}" for i in c3.index], c3.values, color=["#1f3a60", "#2e7d32", "#c85a32"])
axes[1].set_title("(b) Raw K-Means (k=3)")
for i, v in enumerate(c3.values): axes[1].text(i, v + 0.8, str(v), ha="center", fontsize=8.5)
cl2 = pd.Series(lbl_log_k2).value_counts().sort_index()
axes[2].bar([f"C{i}" for i in cl2.index], cl2.values, color=["#1f3a60", "#2e7d32"])
axes[2].set_title("(c) Log1p K-Means (k=2)")
for i, v in enumerate(cl2.values): axes[2].text(i, v + 0.8, str(v), ha="center", fontsize=8.5)
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig3_cluster_sizes.png", dpi=220); plt.close()

fig, ax = plt.subplots(figsize=(8.6, 4.2))
dendrogram(linkage(X_scaled, method="ward"), labels=df["State/UT"].tolist(), leaf_rotation=90, leaf_font_size=7.5, ax=ax, color_threshold=7.0)
ax.set_ylabel("Ward linkage distance"); ax.set_title("Ward-linkage hierarchical clustering dendrogram (32 States/UTs)")
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig4_dendrogram.png", dpi=220); plt.close()

fig, ax = plt.subplots(figsize=(6.4, 3.6))
highlight_states = ["Uttar Pradesh", "Madhya Pradesh", "Rajasthan", "Bihar", "Assam"]
palette = ["#c85a32", "#1f3a60", "#2e7d32", "#6a1b9a", "#00838f"]
for _, row in df.iterrows():
    st = row["State/UT"]
    vals = pd.to_numeric(row[year_columns], errors="coerce").to_numpy(dtype=float)
    if st not in highlight_states and np.isfinite(vals).sum() >= 2:
        ax.plot(short_years, vals, color="#cccccc", linewidth=0.9, alpha=0.7)
for st, col in zip(highlight_states, palette):
    row = df[df["State/UT"] == st].iloc[0]
    vals = pd.to_numeric(row[year_columns], errors="coerce").to_numpy(dtype=float)
    ax.plot(short_years, vals, marker="o", linewidth=1.9, color=col, label=st)
ax.set_xlabel("Financial year"); ax.set_ylabel("Completed road length (km)")
ax.set_title("State-wise PMGSY road-completion trajectories (observed values)")
ax.legend(frameon=False, loc="upper right", fontsize=8.2)
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/fig6_state_trends.png", dpi=220); plt.close()
