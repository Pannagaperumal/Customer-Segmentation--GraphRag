from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..common.utils import data_dir, ensure_directory


FEATURE_EXCLUDE = {
    "muid",
    "location",
    "last_active_at",
    "first_activity",
    "last_activity",
}


def _select_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in FEATURE_EXCLUDE]
    # Keep booleans and numerics; drop non-numeric leftover
    numeric_cols = df[cols].select_dtypes(include=[np.number, bool]).columns.tolist()
    return numeric_cols


def _summarize_cluster(df_users: pd.DataFrame) -> str:
    n = len(df_users)
    if n == 0:
        return "Empty cluster"
    avg_events = df_users["event_count"].mean()
    avg_sessions = df_users["session_count"].mean()
    avg_unique_products = df_users["unique_products"].mean()
    # Device shares
    device_cols = [c for c in df_users.columns if c.startswith("device_") and c.endswith("_share")]
    device_means = df_users[device_cols].mean().sort_values(ascending=False)
    top_devices = ", ".join([f"{col.replace('device_', '').replace('_share', '')}: {val:.1%}" for col, val in device_means.head(3).items()])
    # Category shares (optional if present)
    category_cols = [c for c in df_users.columns if c.startswith("category_") and c.endswith("_share")]
    top_categories = ""
    if category_cols:
        cat_means = df_users[category_cols].mean().sort_values(ascending=False)
        top_categories = ", top_categories: " + ", ".join([
            f"{col.replace('category_', '').replace('_share', '')}: {val:.1%}" for col, val in cat_means.head(3).items()
        ])
    # Funnel flags
    pct_viewed_not_added = df_users["viewed_not_added"].mean()
    pct_added_not_purchased = df_users["added_not_purchased"].mean()
    pct_checkout_abandoned = df_users["checkout_abandoned"].mean()

    summary = (
        f"Cluster of {n} users. Avg events: {avg_events:.1f}, sessions: {avg_sessions:.1f}, "
        f"unique_products: {avg_unique_products:.1f}. Device mix: {top_devices}{top_categories}. "
        f"Funnel flags — viewed_not_added: {pct_viewed_not_added:.1%}, added_not_purchased: {pct_added_not_purchased:.1%}, "
        f"checkout_abandoned: {pct_checkout_abandoned:.1%}."
    )
    return summary


def cluster_users(k: int = 50, random_state: int = 42) -> pd.DataFrame:
    features_path = data_dir("processed", "user_features.parquet")
    df = pd.read_parquet(features_path)

    feature_cols = _select_feature_columns(df)
    X = df[feature_cols].copy()
    # Convert bool to int
    bool_cols = X.select_dtypes(include=[bool]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    cluster_ids = model.fit_predict(X_scaled)

    df_out = df.copy()
    df_out["cluster_id"] = cluster_ids

    # Save assignments
    assignments_path = data_dir("processed", "user_cluster_assignments.parquet")
    df_out[["muid", "cluster_id"]].to_parquet(assignments_path, index=False)

    # Aggregate per cluster
    clusters = []
    for cid, group in df_out.groupby("cluster_id"):
        size = len(group)
        sample_muids = group["muid"].sample(n=min(10, size), random_state=42).tolist()
        summary_text = _summarize_cluster(group)
        clusters.append({
            "cluster_id": int(cid),
            "size": int(size),
            "sample_muids": sample_muids,
            "summary_text": summary_text,
        })

    clusters_df = pd.DataFrame(clusters).sort_values("cluster_id").reset_index(drop=True)
    clusters_path = data_dir("processed", "clusters.parquet")
    clusters_df.to_parquet(clusters_path, index=False)

    return clusters_df


def main() -> None:
    clusters_df = cluster_users()
    print("Saved clusters to", data_dir("processed", "clusters.parquet"))


if __name__ == "__main__":
    main()


