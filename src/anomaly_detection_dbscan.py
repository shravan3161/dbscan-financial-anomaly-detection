"""
DBSCAN Anomaly Detection in Financial Transactions
--------------------------------------------------
Unsupervised Pattern Recognition for Credit Card Fraud Detection
Author: Shravan Lingayat
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, confusion_matrix


def load_data(file_path: str, sample_size: int = 10000, random_state: int = 5) -> pd.DataFrame:
    """Load credit card transactions dataset and return fixed experimental subset."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Total dataset shape: {df.shape}")
    
    if len(df) > sample_size:
        print(f"Sampling {sample_size} records with random_state={random_state}...")
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    
    fraud_count = df['Class'].sum()
    fraud_pct = (fraud_count / len(df)) * 100
    print(f"Working subset: {len(df)} transactions, {fraud_count} fraud cases ({fraud_pct:.3f}%)")
    return df


def preprocess_and_reduce(df: pd.DataFrame, n_components: int = 23):
    """
    Preprocess features using RobustScaler and reduce dimensions with PCA.
    Note: Target 'Class' is separated and excluded from any model fitting.
    """
    X = df.drop(columns=['Class'])
    y = df['Class'].values

    # Step 1: Scale using RobustScaler to handle extreme financial outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 2: Dimensionality reduction using PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_retained = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA reduced dimensions from {X.shape[1]} to {n_components} (Variance retained: {var_retained:.2f}%)")

    return X_pca, y, pca


def compute_k_distance(X: np.ndarray, k: int = 10, output_dir: str = "visualizations"):
    """Compute and plot sorted k-distance graph to guide selection of eps."""
    os.makedirs(output_dir, exist_ok=True)
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    sorted_distances = np.sort(distances[:, k - 1])

    # Key percentiles
    p50 = np.percentile(sorted_distances, 50)
    p95 = np.percentile(sorted_distances, 95)
    p99 = np.percentile(sorted_distances, 99)

    print(f"K-Distance Statistics (k={k}): Median={p50:.4f}, 95th={p95:.4f}, 99th={p99:.4f}")

    plt.figure(figsize=(9, 5))
    plt.plot(sorted_distances, color="#1f77b4", linewidth=2, label=f"{k}-th Nearest Neighbor Distance")
    plt.axhline(y=8.0, color="#d62728", linestyle="--", label="Baseline eps = 8.0")
    plt.axhline(y=6.0, color="#2ca02c", linestyle=":", label="Lower Bound eps = 6.0")
    plt.axhline(y=9.0, color="#ff7f0e", linestyle=":", label="Upper Bound eps = 9.0")
    plt.title(f"Sorted K-Distance Graph (k={k}) for DBSCAN eps Selection", fontsize=12, fontweight="bold")
    plt.xlabel("Points sorted by distance", fontsize=10)
    plt.ylabel(f"Distance to {k}-th Nearest Neighbor", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "k_distance_plot.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved K-distance curve plot to: {save_path}")


def evaluate_dbscan(X: np.ndarray, y: np.ndarray, eps: float, min_samples: int):
    """Fit DBSCAN, treat noise (-1) as anomalies, and compute evaluation metrics."""
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    anomalies_mask = (labels == -1)
    n_anomalies = int(anomalies_mask.sum())
    anomaly_pct = (n_anomalies / len(y)) * 100

    # Binary evaluation: Anomaly (1) vs Normal (0)
    y_pred = anomalies_mask.astype(int)
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "eps": eps,
        "min_samples": min_samples,
        "clusters": n_clusters,
        "anomalies": n_anomalies,
        "anomaly_pct": anomaly_pct,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "labels": labels
    }


def plot_clusters(X_pca: np.ndarray, y: np.ndarray, labels: np.ndarray, output_dir: str = "visualizations"):
    """Plot 2D PCA scatter plot showing normal points, DBSCAN anomalies, and true fraud."""
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))

    is_noise = (labels == -1)
    is_fraud = (y == 1)

    # Plot normal points
    plt.scatter(
        X_pca[~is_noise & ~is_fraud, 0],
        X_pca[~is_noise & ~is_fraud, 1],
        c="#aec7e8",
        alpha=0.4,
        s=15,
        label="Normal (In-Cluster)"
    )

    # Plot false positives (unsupervised anomalies that are not labeled fraud)
    plt.scatter(
        X_pca[is_noise & ~is_fraud, 0],
        X_pca[is_noise & ~is_fraud, 1],
        c="#ff7f0e",
        alpha=0.7,
        s=30,
        edgecolors="k",
        linewidth=0.5,
        label="DBSCAN Anomaly (Legitimate outlier / FP)"
    )

    # Plot true fraud (detected vs missed)
    plt.scatter(
        X_pca[is_noise & is_fraud, 0],
        X_pca[is_noise & is_fraud, 1],
        c="#d62728",
        marker="X",
        s=90,
        edgecolors="k",
        label="True Positive (Fraud detected by DBSCAN)"
    )

    if (~is_noise & is_fraud).sum() > 0:
        plt.scatter(
            X_pca[~is_noise & is_fraud, 0],
            X_pca[~is_noise & is_fraud, 1],
            c="#9467bd",
            marker="^",
            s=90,
            edgecolors="k",
            label="False Negative (Fraud missed by DBSCAN)"
        )

    plt.title("PCA Projection (PC1 vs PC2): DBSCAN Clusters vs Fraud Outliers", fontsize=12, fontweight="bold")
    plt.xlabel("Principal Component 1", fontsize=10)
    plt.ylabel("Principal Component 2", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "pca_clustering_anomaly_scatter.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved clustering scatter plot to: {save_path}")


def run_parameter_sensitivity(X: np.ndarray, y: np.ndarray, output_dir: str = "visualizations"):
    """Evaluate grid of eps and min_samples configurations."""
    eps_values = [6.0, 7.0, 8.0, 9.0]
    min_samples_values = [5, 10, 15]

    results = []
    print("\n--- Running Parameter Sensitivity Analysis ---")
    print(f"{'eps':<6} {'min_samples':<12} {'Clusters':<10} {'Anomalies':<12} {'Recall':<10} {'Precision':<12} {'F1-Score':<10}")
    print("-" * 75)

    for eps in eps_values:
        for ms in min_samples_values:
            res = evaluate_dbscan(X, y, eps=eps, min_samples=ms)
            results.append(res)
            print(f"{res['eps']:<6.1f} {res['min_samples']:<12} {res['clusters']:<10} {res['anomalies']:<12} {res['recall']*100:<9.2f}% {res['precision']*100:<11.2f}% {res['f1_score']:<10.4f}")

    df_results = pd.DataFrame(results)

    # Plot parameter comparison
    os.makedirs(output_dir, exist_ok=True)
    pivot_recall = df_results.pivot(index="eps", columns="min_samples", values="recall")
    pivot_precision = df_results.pivot(index="eps", columns="min_samples", values="precision")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(pivot_recall * 100, annot=True, fmt=".1f", cmap="Blues", ax=axes[0], cbar_kws={'label': 'Recall (%)'})
    axes[0].set_title("Recall across eps & min_samples (%)", fontweight="bold")

    sns.heatmap(pivot_precision * 100, annot=True, fmt=".2f", cmap="Greens", ax=axes[1], cbar_kws={'label': 'Precision (%)'})
    axes[1].set_title("Precision across eps & min_samples (%)", fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "parameter_tuning_heatmap.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved parameter sensitivity heatmap to: {save_path}")

    return df_results


def main():
    parser = argparse.ArgumentParser(description="DBSCAN Financial Transaction Anomaly Detection")
    parser.add_argument("--data_path", type=str, default="data/sample_creditcard_10k.csv", help="Path to CSV dataset")
    parser.add_argument("--sample_size", type=int, default=10000, help="Evaluation sample size")
    parser.add_argument("--eps", type=float, default=8.0, help="DBSCAN eps neighborhood radius")
    parser.add_argument("--min_samples", type=int, default=5, help="DBSCAN min_samples threshold (best config=5, baseline=10)")
    parser.add_argument("--grid_search", action="store_true", help="Run full parameter sensitivity grid")
    args = parser.parse_args()

    # Step 1: Ingestion
    df = load_data(args.data_path, sample_size=args.sample_size)

    # Step 2: Scaling and PCA
    X_pca, y, _ = preprocess_and_reduce(df, n_components=23)

    # Step 3: K-Distance Elbow Plot
    compute_k_distance(X_pca, k=10)

    # Step 4: Model Evaluation
    print(f"\n--- Evaluating Configuration (eps={args.eps}, min_samples={args.min_samples}) ---")
    best_res = evaluate_dbscan(X_pca, y, eps=args.eps, min_samples=args.min_samples)
    print(f"Clusters Identified : {best_res['clusters']}")
    print(f"Anomalies Flagged   : {best_res['anomalies']} ({best_res['anomaly_pct']:.2f}% of transactions)")
    print(f"True Positives (TP) : {best_res['tp']} / {best_res['tp'] + best_res['fn']} fraud cases captured")
    print(f"False Positives (FP): {best_res['fp']}")
    print(f"Recall              : {best_res['recall']*100:.2f}%")
    print(f"Precision           : {best_res['precision']*100:.2f}%")
    print(f"F1-Score            : {best_res['f1_score']:.4f}")

    # Step 5: Visualizations
    plot_clusters(X_pca, y, best_res["labels"])

    # Step 6: Parameter Grid (optional or default)
    if args.grid_search:
        run_parameter_sensitivity(X_pca, y)

    print("\n[SUCCESS] Pipeline executed successfully. Visualizations stored in 'visualizations/' directory.")


if __name__ == "__main__":
    main()
