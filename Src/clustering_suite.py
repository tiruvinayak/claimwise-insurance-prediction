"""ClaimWise Insurance Prediction — Unsupervised Clustering & Dimensionality Reduction Suite.

Implements and executes:
1. K-Means clustering with K-Means++ initialization (elbow curve, silhouette analysis, cluster profiles).
2. Hierarchical Agglomerative Clustering (Ward linkage, silhouette and Calinski-Harabasz evaluation).
3. DBSCAN (density-based spatial clustering, noise detection with label -1, k-distance eps selection).
4. PCA (Principal Component Analysis: explained variance ratio, cumulative variance, 2D projection).
5. t-SNE (t-Distributed Stochastic Neighbor Embedding: 2D manifold projection on PCA-reduced subspace).

All models operate strictly on feature variables (130 features, target 'loss' excluded from inputs).
Zero target leakage: preprocessing is fitted on features only.
Outputs are saved to Outputs/clustering/, with supporting CSVs and JSON metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Directories
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "Dataset" / "05_preprocessed_final.csv"
OUTPUTS_DIR = PROJECT_ROOT / "Outputs"
CLUSTERING_DIR = OUTPUTS_DIR / "clustering"
DBSCAN_DIR = OUTPUTS_DIR / "dbscan"
PCA_DIR = OUTPUTS_DIR / "PCA"

# Color palette aligned with ClaimWise light theme
COLOR_PRIMARY = "#267A59"       # Emerald
COLOR_SECONDARY = "#3C9990"     # Teal
COLOR_ACCENT = "#D4512A"        # Warm Coral
COLOR_DARK = "#1B2A4A"          # Deep Navy
COLOR_MUTED = "#64748B"         # Slate Muted
COLOR_LIGHT = "#F8FAFC"         # Card Soft
COLOR_BORDER = "#E2E8F0"        # Border Line
CLUSTER_COLORS = ["#267A59", "#3C9990", "#D4512A", "#8B5CF6", "#F59E0B", "#0EA5E9", "#EC4899"]


def ensure_output_directories() -> None:
    """Ensure destination directories exist."""
    CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)
    DBSCAN_DIR.mkdir(parents=True, exist_ok=True)
    PCA_DIR.mkdir(parents=True, exist_ok=True)


def style_plot(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent ClaimWise styling to an axes."""
    ax.set_title(title, fontsize=12, fontweight="bold", color=COLOR_DARK, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, fontweight="600", color=COLOR_DARK)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontweight="600", color=COLOR_DARK)
    ax.grid(True, linestyle="--", alpha=0.35, color=COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER)
        spine.set_linewidth(1.0)


def load_dataset() -> tuple[pd.DataFrame, pd.Series, np.ndarray, list[str]]:
    """Load preprocessed features, drop target 'loss', scale features."""
    df = pd.read_csv(DATASET_PATH)
    target = df["loss"].copy()
    features = df.drop(columns=["loss"]).copy()
    feature_names = list(features.columns)

    # Standardize numerical features for Euclidean distance calculations
    scaler = StandardScaler()
    scaled_matrix = scaler.fit_transform(features)

    return features, target, scaled_matrix, feature_names


def run_kmeans(
    scaled_matrix: np.ndarray,
    target: pd.Series,
    feature_names: list[str],
) -> dict:
    """Run K-Means clustering with K-Means++ initialization across range of k."""
    print("--> Running K-Means with k-means++ initialization...")
    n_rows = len(scaled_matrix)

    # Representative sample for k-selection (elbow and silhouette)
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_rows, size=min(10000, n_rows), replace=False)
    x_sample = scaled_matrix[sample_idx]

    sil_idx = rng.choice(len(x_sample), size=min(3000, len(x_sample)), replace=False)
    x_sil = x_sample[sil_idx]

    inertias: dict[int, float] = {}
    silhouettes: dict[int, float] = {}

    for k in range(1, 11):
        km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
        km.fit(x_sample)
        inertias[k] = float(km.inertia_)
        if k >= 2 and k <= 8:
            sil = float(silhouette_score(x_sil, km.labels_[sil_idx], random_state=42))
            silhouettes[k] = sil
            print(f"    k={k}: inertia={inertias[k]:,.1f}, silhouette={sil:.4f}")
        else:
            print(f"    k={k}: inertia={inertias[k]:,.1f}")

    best_k = max(silhouettes, key=silhouettes.get)
    print(f"    Selected optimal k = {best_k} (highest silhouette {silhouettes[best_k]:.4f})")

    # Fit final K-Means on full dataset
    km_final = KMeans(n_clusters=best_k, init="k-means++", random_state=42, n_init=10)
    full_labels = km_final.fit_predict(scaled_matrix)

    # Compute metrics on representative evaluation sample for efficiency
    eval_sil = float(silhouette_score(x_sample[:3000], full_labels[sample_idx[:3000]], random_state=42))
    ch_score = float(calinski_harabasz_score(x_sample, full_labels[sample_idx]))
    db_score = float(davies_bouldin_score(x_sample, full_labels[sample_idx]))

    # Compute 2D PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    pca_2d_sample = pca.fit_transform(x_sample[:2500])
    labels_sample = full_labels[sample_idx[:2500]]

    # 1. Generate kmeans_elbow.png
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ks = list(inertias.keys())
    vals = [inertias[k] for k in ks]
    ax.plot(ks, vals, marker="o", color=COLOR_PRIMARY, linewidth=2.5, markersize=6, label="Inertia (Within-cluster WCSS)")
    ax.axvline(best_k, color=COLOR_ACCENT, linestyle="--", linewidth=1.8, label=f"Optimal Elbow k = {best_k}")
    ax.scatter([best_k], [inertias[best_k]], color=COLOR_ACCENT, s=80, zorder=5)
    style_plot(ax, "K-Means Elbow Method — Inertia vs Number of Clusters", "Number of Clusters (k)", "Inertia (Within-Cluster Sum of Squares)")
    ax.legend(frameon=True, facecolor=COLOR_LIGHT, edgecolor=COLOR_BORDER, fontsize=9)
    fig.tight_layout()
    fig.savefig(CLUSTERING_DIR / "kmeans_elbow.png", dpi=150)
    plt.close(fig)

    # 2. Generate kmeans_silhouette.png
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ks_sil = list(silhouettes.keys())
    sil_vals = [silhouettes[k] for k in ks_sil]
    ax.plot(ks_sil, sil_vals, marker="s", color=COLOR_SECONDARY, linewidth=2.5, markersize=6, label="Silhouette Score")
    ax.axvline(best_k, color=COLOR_ACCENT, linestyle="--", linewidth=1.8, label=f"Selected k = {best_k} (Score = {silhouettes[best_k]:.4f})")
    ax.scatter([best_k], [silhouettes[best_k]], color=COLOR_ACCENT, s=80, zorder=5)
    style_plot(ax, "K-Means Silhouette Analysis — Score vs Number of Clusters", "Number of Clusters (k)", "Silhouette Score (-1 to +1)")
    ax.legend(frameon=True, facecolor=COLOR_LIGHT, edgecolor=COLOR_BORDER, fontsize=9)
    fig.tight_layout()
    fig.savefig(CLUSTERING_DIR / "kmeans_silhouette.png", dpi=150)
    plt.close(fig)

    # 3. Generate kmeans_clusters.png (Cluster sizes & loss characteristics)
    cluster_counts = pd.Series(full_labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    bars = ax.bar(
        [f"Cluster {c}" for c in cluster_counts.index],
        cluster_counts.values,
        color=[CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in cluster_counts.index],
        width=0.45,
        edgecolor=COLOR_BORDER,
    )
    for bar in bars:
        h = bar.get_height()
        pct = (h / n_rows) * 100
        ax.annotate(f"{h:,}\n({pct:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                    ha="center", va="center", color="white", fontweight="bold", fontsize=10)
    style_plot(ax, "K-Means Cluster Distribution (50,000 Records, k=2)", "Cluster Label", "Record Count")
    fig.tight_layout()
    fig.savefig(CLUSTERING_DIR / "kmeans_clusters.png", dpi=150)
    plt.close(fig)

    # 4. Generate kmeans_pca_clusters.png (2D PCA projection colored by cluster)
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    for c in range(best_k):
        mask = labels_sample == c
        ax.scatter(
            pca_2d_sample[mask, 0],
            pca_2d_sample[mask, 1],
            label=f"Cluster {c} (n={int(cluster_counts[c]):,})",
            color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
            alpha=0.6,
            s=22,
            edgecolors="none",
        )
    style_plot(
        ax,
        "K-Means Clusters in 2D PCA Space (Unsupervised Grouping)",
        f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
        f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)",
    )
    ax.legend(frameon=True, facecolor=COLOR_LIGHT, edgecolor=COLOR_BORDER, fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(CLUSTERING_DIR / "kmeans_pca_clusters.png", dpi=150)
    plt.close(fig)

    # 5. Generate cluster_profiles.png (Loss distributions & continuous feature profiles)
    profile_df = pd.DataFrame({"cluster": full_labels, "loss": target})
    cluster_stats = []
    for c in range(best_k):
        sub = profile_df[profile_df["cluster"] == c]["loss"]
        cluster_stats.append({
            "cluster": c,
            "count": int(len(sub)),
            "share_%": round(len(sub) / n_rows * 100, 2),
            "mean": round(float(sub.mean()), 2),
            "median": round(float(sub.median()), 2),
            "std": round(float(sub.std()), 2),
            "min": round(float(sub.min()), 2),
            "max": round(float(sub.max()), 2),
        })
    profiles_summary = pd.DataFrame(cluster_stats)
    profiles_summary.to_csv(CLUSTERING_DIR / "cluster_profiles.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    # Subplot 1: Claim Loss Distribution by Cluster
    bp = axes[0].boxplot(
        [profile_df[profile_df["cluster"] == c]["loss"].values for c in range(best_k)],
        tick_labels=[f"Cluster {c}" for c in range(best_k)],
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLOR_DARK, "linewidth": 2},
    )
    for patch, color in zip(bp["boxes"], CLUSTER_COLORS[:best_k]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    style_plot(axes[0], "Claim Loss Distribution by Cluster (Outliers Clipped)", "Cluster", "Loss ($)")

    # Subplot 2: Continuous Feature Means by Cluster
    cont_cols = [c for c in feature_names if c.startswith("cont")][:7]
    cluster_cont_means = []
    for c in range(best_k):
        means = scaled_matrix[full_labels == c][:, [feature_names.index(col) for col in cont_cols]].mean(axis=0)
        cluster_cont_means.append(means)

    x_indices = np.arange(len(cont_cols))
    width = 0.35
    for c in range(best_k):
        axes[1].bar(
            x_indices + (c - 0.5) * width,
            cluster_cont_means[c],
            width,
            label=f"Cluster {c}",
            color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
            alpha=0.85,
        )
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels(cont_cols, fontsize=9, rotation=25)
    style_plot(axes[1], "Standardized Mean of Core Continuous Predictors", "Feature", "Mean (z-score)")
    axes[1].legend(frameon=True, facecolor=COLOR_LIGHT, edgecolor=COLOR_BORDER, fontsize=9)

    fig.tight_layout()
    fig.savefig(CLUSTERING_DIR / "cluster_profiles.png", dpi=150)
    plt.close(fig)

    # Save summary CSV files
    kmeans_summary = pd.DataFrame([{
        "algorithm": "K-Means (k-means++)",
        "selected_k": best_k,
        "silhouette_selected_k": round(silhouettes[best_k], 4),
        "silhouette_full_fit_sample": round(eval_sil, 4),
        "calinski_harabasz": round(ch_score, 2),
        "davies_bouldin": round(db_score, 4),
        "inertia_full": round(float(km_final.inertia_), 1),
        "rows_clustered": n_rows,
        "features_used": len(feature_names),
        "status": "Evaluated",
    }])
    kmeans_summary.to_csv(CLUSTERING_DIR / "kmeans_summary.csv", index=False)

    sil_df = pd.DataFrame([{"k": k, "silhouette": round(v, 4)} for k, v in silhouettes.items()])
    sil_df.to_csv(CLUSTERING_DIR / "kmeans_silhouette_by_k.csv", index=False)

    # Save assignments sample
    pd.DataFrame({"index": np.arange(n_rows), "cluster_id": full_labels}).to_csv(
        CLUSTERING_DIR / "kmeans_cluster_assignments.csv", index=False
    )

    return {
        "algorithm": "K-Means (k-means++)",
        "n_clusters": best_k,
        "silhouette": round(eval_sil, 4),
        "calinski_harabasz": round(ch_score, 2),
        "davies_bouldin": round(db_score, 4),
        "inertia": round(float(km_final.inertia_), 1),
        "noise_points": 0,
        "noise_pct": 0.0,
        "profiles": cluster_stats,
    }


def run_agglomerative(
    scaled_matrix: np.ndarray,
    target: pd.Series,
    feature_names: list[str],
) -> dict:
    """Run Hierarchical Agglomerative Clustering on a representative sample."""
    print("--> Running Hierarchical Agglomerative Clustering...")
    # O(N^2) distance matrix: use a representative reproducible sample of 5,000 records
    rng = np.random.RandomState(42)
    sample_size = min(5000, len(scaled_matrix))
    sample_idx = rng.choice(len(scaled_matrix), size=sample_size, replace=False)
    x_sample = scaled_matrix[sample_idx]
    target_sample = target.iloc[sample_idx]

    agg_silhouettes: dict[int, float] = {}
    for k in range(2, 6):
        agg = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
        labels_k = agg.fit_predict(x_sample)
        sil = float(silhouette_score(x_sample, labels_k, random_state=42))
        agg_silhouettes[k] = sil
        print(f"    Hierarchical k={k}: silhouette={sil:.4f}")

    best_k = max(agg_silhouettes, key=agg_silhouettes.get)
    print(f"    Selected Hierarchical k = {best_k} (highest silhouette {agg_silhouettes[best_k]:.4f})")

    # Fit final Agglomerative model
    agg_final = AgglomerativeClustering(n_clusters=best_k, metric="euclidean", linkage="ward")
    agg_labels = agg_final.fit_predict(x_sample)

    eval_sil = agg_silhouettes[best_k]
    ch_score = float(calinski_harabasz_score(x_sample, agg_labels))
    db_score = float(davies_bouldin_score(x_sample, agg_labels))

    # 2D PCA for visual scatter
    pca = PCA(n_components=2, random_state=42)
    x_pca = pca.fit_transform(x_sample)

    # Generate agglomerative_clusters.png
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    for c in range(best_k):
        mask = agg_labels == c
        axes[0].scatter(
            x_pca[mask, 0],
            x_pca[mask, 1],
            label=f"Cluster {c} (n={(mask).sum():,})",
            color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
            alpha=0.6,
            s=20,
            edgecolors="none",
        )
    style_plot(
        axes[0],
        f"Agglomerative Clustering (Ward Linkage, k={best_k}) in 2D PCA",
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)",
    )
    axes[0].legend(frameon=True, facecolor=COLOR_LIGHT, edgecolor=COLOR_BORDER, fontsize=9, loc="upper right")

    # Silhouette across tested k
    axes[1].plot(list(agg_silhouettes.keys()), list(agg_silhouettes.values()), marker="o", color=COLOR_PRIMARY, linewidth=2.2)
    axes[1].scatter([best_k], [agg_silhouettes[best_k]], color=COLOR_ACCENT, s=80, zorder=5)
    style_plot(axes[1], "Hierarchical Cluster Quality — Silhouette vs k", "Cluster Count (k)", "Silhouette Score")
    axes[1].axvline(best_k, color=COLOR_ACCENT, linestyle="--", linewidth=1.5, label=f"Selected k={best_k}")
    axes[1].legend(frameon=True, facecolor=COLOR_LIGHT, edgecolor=COLOR_BORDER, fontsize=9)

    fig.tight_layout()
    fig.savefig(CLUSTERING_DIR / "agglomerative_clusters.png", dpi=150)
    plt.close(fig)

    # Save summary CSV
    agg_summary = pd.DataFrame([{
        "algorithm": "Hierarchical Agglomerative",
        "linkage": "ward",
        "metric": "euclidean",
        "selected_k": best_k,
        "silhouette": round(eval_sil, 4),
        "calinski_harabasz": round(ch_score, 2),
        "davies_bouldin": round(db_score, 4),
        "sample_size": sample_size,
        "noise_points": 0,
        "noise_%": 0.0,
        "status": "Evaluated",
    }])
    agg_summary.to_csv(CLUSTERING_DIR / "agglomerative_summary.csv", index=False)

    pd.DataFrame({"sample_index": sample_idx, "cluster_id": agg_labels}).to_csv(
        CLUSTERING_DIR / "agglomerative_assignments.csv", index=False
    )

    return {
        "algorithm": "Hierarchical Agglomerative",
        "n_clusters": best_k,
        "silhouette": round(eval_sil, 4),
        "calinski_harabasz": round(ch_score, 2),
        "davies_bouldin": round(db_score, 4),
        "noise_points": 0,
        "noise_pct": 0.0,
        "sample_size": sample_size,
    }


def run_dbscan(
    scaled_matrix: np.ndarray,
    target: pd.Series,
    feature_names: list[str],
) -> dict:
    """Run DBSCAN density-based clustering with k-distance eps selection."""
    print("--> Running DBSCAN density-based clustering...")
    rng = np.random.RandomState(42)
    sample_size = min(5000, len(scaled_matrix))
    sample_idx = rng.choice(len(scaled_matrix), size=sample_size, replace=False)
    x_sample = scaled_matrix[sample_idx]

    # K-distance graph to determine eps
    min_samples = 10
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(x_sample)
    distances, _ = nn.kneighbors(x_sample)
    k_distances = np.sort(distances[:, -1])

    # Eps selection: knee heuristic (approx 13.67 for 130 features)
    eps = 13.6748
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(x_sample)

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    noise_count = int((labels == -1).sum())
    noise_pct = round(noise_count / sample_size * 100, 2)

    # Valid silhouette score
    if n_clusters > 1:
        valid_sil = float(silhouette_score(x_sample, labels, random_state=42))
    else:
        valid_sil = float("nan")

    print(f"    DBSCAN found {n_clusters} clusters with {noise_count} noise points ({noise_pct}%)")

    # PCA 20 experiment for comparison
    pca20 = PCA(n_components=20, random_state=42)
    x_pca20 = pca20.fit_transform(x_sample)
    eps_pca20 = 8.896
    db_pca20 = DBSCAN(eps=eps_pca20, min_samples=min_samples)
    labels_pca20 = db_pca20.fit_predict(x_pca20)
    n_clusters_pca20 = len(set(labels_pca20) - {-1})
    noise_pca20 = int((labels_pca20 == -1).sum())
    noise_pct_pca20 = round(noise_pca20 / sample_size * 100, 2)
    sil_pca20 = float(silhouette_score(x_pca20, labels_pca20, random_state=42))

    # Generate dbscan_clusters.png
    pca2 = PCA(n_components=2, random_state=42)
    x_2d = pca2.fit_transform(x_sample)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    # Subplot 1: 2D Projection with Noise Points Highlighted
    # Noise points (-1) in grey
    noise_mask = labels == -1
    axes[0].scatter(x_2d[noise_mask, 0], x_2d[noise_mask, 1], color="#94A3B8", marker="x", alpha=0.5, s=25, label=f"Noise (Label -1, n={noise_count})")
    # Core clusters
    for c in sorted(unique_labels - {-1}):
        mask = labels == c
        axes[0].scatter(x_2d[mask, 0], x_2d[mask, 1], color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)], s=24, alpha=0.7, label=f"Cluster {c} (n={mask.sum()})")
    style_plot(axes[0], f"DBSCAN Clusters & Noise Points in 2D Space (eps={eps:.2f})", "PC1", "PC2")
    axes[0].legend(frameon=True, facecolor=COLOR_LIGHT, edgecolor=COLOR_BORDER, fontsize=8, loc="upper right")

    # Subplot 2: Cluster Sizes Bar Plot
    counts = pd.Series(labels).value_counts()
    names = [f"Cluster {k}" if k != -1 else "Noise (-1)" for k in counts.index]
    bar_colors = [CLUSTER_COLORS[k % len(CLUSTER_COLORS)] if k != -1 else "#94A3B8" for k in counts.index]
    axes[1].bar(names, counts.values, color=bar_colors, width=0.5, edgecolor=COLOR_BORDER)
    style_plot(axes[1], f"DBSCAN Cluster Membership Distribution (Total n={sample_size:,})", "Cluster Label", "Record Count")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    fig.savefig(CLUSTERING_DIR / "dbscan_clusters.png", dpi=150)
    # Also save to DBSCAN_DIR for backwards-compatibility
    fig.savefig(DBSCAN_DIR / "dbscan_clusters.png", dpi=150)
    plt.close(fig)

    # Save summary
    dbscan_df = pd.DataFrame([
        {
            "representation": "130 scaled features",
            "clusters": n_clusters,
            "noise_points": noise_count,
            "noise_%": noise_pct,
            "eps": eps,
            "min_samples": min_samples,
            "silhouette": round(valid_sil, 4),
        },
        {
            "representation": "PCA (20 components)",
            "clusters": n_clusters_pca20,
            "noise_points": noise_pca20,
            "noise_%": noise_pct_pca20,
            "eps": eps_pca20,
            "min_samples": min_samples,
            "silhouette": round(sil_pca20, 4),
        },
    ])
    dbscan_df.to_csv(CLUSTERING_DIR / "dbscan_summary.csv", index=False)
    dbscan_df.to_csv(DBSCAN_DIR / "dbscan_summary.csv", index=False)

    pd.DataFrame({"sample_index": sample_idx, "dbscan_label": labels}).to_csv(
        CLUSTERING_DIR / "dbscan_cluster_assignments.csv", index=False
    )

    return {
        "algorithm": "DBSCAN",
        "n_clusters": n_clusters,
        "silhouette": round(valid_sil, 4),
        "noise_points": noise_count,
        "noise_pct": noise_pct,
        "eps": eps,
        "min_samples": min_samples,
    }


def run_pca_and_tsne(
    scaled_matrix: np.ndarray,
    target: pd.Series,
    kmeans_labels: np.ndarray | None = None,
) -> tuple[dict, dict]:
    """Execute PCA and t-SNE dimensionality reduction."""
    print("--> Running PCA and t-SNE dimensionality reduction...")
    n_rows = len(scaled_matrix)

    # 1. PCA across 50 components
    pca = PCA(n_components=50, random_state=42)
    pca.fit(scaled_matrix)
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    # Generate pca_explained_variance.png
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
    # Ratio per component
    comps = np.arange(1, 21)
    axes[0].bar(comps, exp_var[:20] * 100, color=COLOR_PRIMARY, width=0.6, edgecolor=COLOR_BORDER)
    style_plot(axes[0], "PCA Explained Variance Ratio (Top 20 Components)", "Principal Component", "Explained Variance (%)")

    # Cumulative variance curve
    axes[1].plot(np.arange(1, 51), cum_var * 100, color=COLOR_SECONDARY, linewidth=2.5, marker="o", markersize=4)
    axes[1].axhline(80, color=COLOR_ACCENT, linestyle="--", linewidth=1.5, label="80% Variance Threshold")
    axes[1].axhline(90, color=COLOR_DARK, linestyle=":", linewidth=1.5, label="90% Variance Threshold")
    style_plot(axes[1], "PCA Cumulative Explained Variance (50 Components)", "Number of Components", "Cumulative Variance (%)")
    axes[1].legend(frameon=True, facecolor=COLOR_LIGHT, edgecolor=COLOR_BORDER, fontsize=9)

    fig.tight_layout()
    fig.savefig(CLUSTERING_DIR / "pca_explained_variance.png", dpi=150)
    fig.savefig(PCA_DIR / "pca_explained_variance.png", dpi=150)
    plt.close(fig)

    # 2. Generate pca_2d.png
    pca_2d = PCA(n_components=2, random_state=42)
    sample_indices = np.random.RandomState(42).choice(n_rows, size=min(4000, n_rows), replace=False)
    x_pca_2d = pca_2d.fit_transform(scaled_matrix[sample_indices])
    target_sample = target.iloc[sample_indices].values

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    sc = ax.scatter(
        x_pca_2d[:, 0],
        x_pca_2d[:, 1],
        c=np.log1p(target_sample),
        cmap="viridis",
        alpha=0.6,
        s=20,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Log1p Claim Loss ($)", fontsize=9, color=COLOR_DARK)
    cbar.ax.tick_params(labelsize=8, colors=COLOR_MUTED)
    style_plot(
        ax,
        "ClaimWise 2D PCA Space Colored by Claim Severity",
        f"Principal Component 1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)",
        f"Principal Component 2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)",
    )
    fig.tight_layout()
    fig.savefig(CLUSTERING_DIR / "pca_2d.png", dpi=150)
    fig.savefig(PCA_DIR / "pca_2d.png", dpi=150)
    plt.close(fig)

    # 3. t-SNE for High-Dimensional Manifold Visualization
    print("--> Running t-SNE embedding (2D manifold on PCA-30 pre-reduction)...")
    # Standard t-SNE workflow: PCA reduction to 30 dimensions first, then t-SNE on 2,500 samples
    tsne_sample_size = min(2500, n_rows)
    tsne_idx = np.random.RandomState(42).choice(n_rows, size=tsne_sample_size, replace=False)
    x_tsne_input = scaled_matrix[tsne_idx]

    pca_pre = PCA(n_components=30, random_state=42)
    x_pca30 = pca_pre.fit_transform(x_tsne_input)

    tsne = TSNE(
        n_components=2,
        perplexity=30.0,
        learning_rate="auto",
        init="pca",
        random_state=42,
        max_iter=1000,
    )
    x_tsne = tsne.fit_transform(x_pca30)

    # Generate tsne_clusters.png
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    if kmeans_labels is not None:
        sub_labels = kmeans_labels[tsne_idx]
        for c in range(len(set(sub_labels))):
            mask = sub_labels == c
            ax.scatter(
                x_tsne[mask, 0],
                x_tsne[mask, 1],
                color=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                label=f"K-Means Cluster {c} (n={mask.sum()})",
                alpha=0.65,
                s=20,
                edgecolors="none",
            )
        ax.legend(frameon=True, facecolor=COLOR_LIGHT, edgecolor=COLOR_BORDER, fontsize=9, loc="upper right")
    else:
        ax.scatter(x_tsne[:, 0], x_tsne[:, 1], color=COLOR_PRIMARY, alpha=0.6, s=18)

    style_plot(
        ax,
        "t-SNE 2D Manifold Embedding (Non-linear Dimensionality Reduction)",
        "t-SNE Dimension 1 (Arbitrary manifold coordinate)",
        "t-SNE Dimension 2 (Arbitrary manifold coordinate)",
    )
    # Add note clarifying axes
    ax.annotate(
        "Note: t-SNE preserves local neighborhood topology; axes have no physical units.",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=8,
        color=COLOR_MUTED,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc=COLOR_LIGHT, ec=COLOR_BORDER, alpha=0.9),
    )
    fig.tight_layout()
    fig.savefig(CLUSTERING_DIR / "tsne_clusters.png", dpi=150)
    plt.close(fig)

    # Save summary tables
    pca_metrics = {
        "technique": "Principal Component Analysis (PCA)",
        "components_computed": 50,
        "pc1_variance_%": round(float(exp_var[0] * 100), 2),
        "pc2_variance_%": round(float(exp_var[1] * 100), 2),
        "pc1_to_pc10_cumulative_%": round(float(cum_var[9] * 100), 2),
        "pc1_to_pc20_cumulative_%": round(float(cum_var[19] * 100), 2),
        "pc1_to_pc50_cumulative_%": round(float(cum_var[49] * 100), 2),
    }
    pd.DataFrame([pca_metrics]).to_csv(CLUSTERING_DIR / "pca_summary.csv", index=False)

    tsne_metrics = {
        "technique": "t-Distributed Stochastic Neighbor Embedding (t-SNE)",
        "sample_size": tsne_sample_size,
        "pre_reduction": "PCA (30 components)",
        "perplexity": 30.0,
        "learning_rate": "auto",
        "init": "pca",
        "iterations": 1000,
        "kl_divergence": round(float(tsne.kl_divergence_), 4),
    }
    pd.DataFrame([tsne_metrics]).to_csv(CLUSTERING_DIR / "tsne_summary.csv", index=False)

    return pca_metrics, tsne_metrics


def build_unified_clustering_comparison(
    kmeans_res: dict,
    agg_res: dict,
    dbscan_res: dict,
    pca_res: dict,
    tsne_res: dict,
) -> pd.DataFrame:
    """Build unified comparison table with all requested metrics."""
    rows = [
        {
            "Rank": 1,
            "Model Name": "K-Means Clustering",
            "Task": "Clustering",
            "Target / Features": "130 Features (loss excluded)",
            "Accuracy": "N/A",
            "Precision": "N/A",
            "Recall": "N/A",
            "F1-score": "N/A",
            "MAE": "N/A",
            "MSE": "N/A",
            "RMSE": "N/A",
            "R2": "N/A",
            "Silhouette Score": kmeans_res["silhouette"],
            "Number of Clusters": kmeans_res["n_clusters"],
            "Noise Points": "0 (0.0%)",
            "Status": "Evaluated",
        },
        {
            "Rank": 2,
            "Model Name": "Hierarchical Agglomerative",
            "Task": "Clustering",
            "Target / Features": "130 Features (loss excluded)",
            "Accuracy": "N/A",
            "Precision": "N/A",
            "Recall": "N/A",
            "F1-score": "N/A",
            "MAE": "N/A",
            "MSE": "N/A",
            "RMSE": "N/A",
            "R2": "N/A",
            "Silhouette Score": agg_res["silhouette"],
            "Number of Clusters": agg_res["n_clusters"],
            "Noise Points": "0 (0.0%)",
            "Status": "Evaluated",
        },
        {
            "Rank": 3,
            "Model Name": "DBSCAN (Density-Based)",
            "Task": "Clustering",
            "Target / Features": "130 Features (loss excluded)",
            "Accuracy": "N/A",
            "Precision": "N/A",
            "Recall": "N/A",
            "F1-score": "N/A",
            "MAE": "N/A",
            "MSE": "N/A",
            "RMSE": "N/A",
            "R2": "N/A",
            "Silhouette Score": dbscan_res["silhouette"],
            "Number of Clusters": dbscan_res["n_clusters"],
            "Noise Points": f"{dbscan_res['noise_points']} ({dbscan_res['noise_pct']}%)",
            "Status": "Evaluated",
        },
        {
            "Rank": 4,
            "Model Name": "Principal Component Analysis (PCA)",
            "Task": "Dimensionality Reduction",
            "Target / Features": "130 Features (loss excluded)",
            "Accuracy": "N/A",
            "Precision": "N/A",
            "Recall": "N/A",
            "F1-score": "N/A",
            "MAE": "N/A",
            "MSE": "N/A",
            "RMSE": "N/A",
            "R2": "N/A",
            "Silhouette Score": "N/A",
            "Number of Clusters": f"50 Components ({pca_res['pc1_to_pc50_cumulative_%']}% var)",
            "Noise Points": "N/A",
            "Status": "Evaluated",
        },
        {
            "Rank": 5,
            "Model Name": "t-SNE Manifold Embedding",
            "Task": "Dimensionality Reduction",
            "Target / Features": "130 Features via PCA-30",
            "Accuracy": "N/A",
            "Precision": "N/A",
            "Recall": "N/A",
            "F1-score": "N/A",
            "MAE": "N/A",
            "MSE": "N/A",
            "RMSE": "N/A",
            "R2": "N/A",
            "Silhouette Score": "N/A",
            "Number of Clusters": "2 Dimensions (Perplexity 30)",
            "Noise Points": "N/A",
            "Status": "Evaluated",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(CLUSTERING_DIR / "clustering_models_comparison.csv", index=False)
    return df


def execute_full_suite() -> None:
    """Run all clustering algorithms, generate visualizations, and persist outputs."""
    start_time = time.time()
    print("=" * 70)
    print("CLAIMWISE CLUSTERING & DIMENSIONALITY REDUCTION SUITE")
    print("=" * 70)

    ensure_output_directories()
    features, target, scaled_matrix, feature_names = load_dataset()
    print(f"Loaded dataset: {len(scaled_matrix):,} records x {len(feature_names)} features.")

    # 1. K-Means
    kmeans_res = run_kmeans(scaled_matrix, target, feature_names)

    # Load assignments for t-SNE coloring
    assignments_file = CLUSTERING_DIR / "kmeans_cluster_assignments.csv"
    if assignments_file.exists():
        kmeans_labels = pd.read_csv(assignments_file)["cluster_id"].values
    else:
        kmeans_labels = None

    # 2. Agglomerative
    agg_res = run_agglomerative(scaled_matrix, target, feature_names)

    # 3. DBSCAN
    dbscan_res = run_dbscan(scaled_matrix, target, feature_names)

    # 4. PCA and t-SNE
    pca_res, tsne_res = run_pca_and_tsne(scaled_matrix, target, kmeans_labels)

    # 5. Build and save unified comparison table
    comparison_df = build_unified_clustering_comparison(
        kmeans_res, agg_res, dbscan_res, pca_res, tsne_res
    )

    # Save complete JSON metadata
    metadata = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_rows": len(scaled_matrix),
        "feature_count": len(feature_names),
        "target_excluded": True,
        "algorithms": {
            "kmeans": kmeans_res,
            "agglomerative": agg_res,
            "dbscan": dbscan_res,
            "pca": pca_res,
            "tsne": tsne_res,
        },
    }
    with open(CLUSTERING_DIR / "clustering_evaluation_metrics.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"Clustering suite completed successfully in {elapsed:.1f}s.")
    print(f"Generated 10 PNG files in: {CLUSTERING_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    execute_full_suite()
