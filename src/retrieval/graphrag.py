from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from neo4j import GraphDatabase, basic_auth
from sentence_transformers import SentenceTransformer

from ..common.utils import data_dir, get_env
from .vector_index import ClusterVectorIndex


DEVICE_PATTERNS = {
    "android": r"\bandroid\b",
    "ios": r"\bios\b|\biphone\b|\bipad\b",
    "windows": r"\bwindows\b",
    "macos": r"\bmacos\b|\bmac\b|\bos x\b",
    "linux": r"\blinux\b",
}

CATEGORY_KEYWORDS = ["electronics", "fashion", "home", "sports", "beauty", "toys"]

FUNNEL_FLAGS = {
    "viewed_not_added": [r"viewed but not added", r"viewed_not_added"],
    "added_not_purchased": [r"added but not purchased", r"added_not_purchased"],
    "checkout_abandoned": [r"checkout abandoned", r"checkout_abandoned"],
}


def parse_filters(query: str) -> Dict[str, List[str]]:
    query_l = query.lower()
    devices = [name for name, pat in DEVICE_PATTERNS.items() if re.search(pat, query_l)]
    categories = [c for c in CATEGORY_KEYWORDS if c in query_l]
    flags = []
    for key, patterns in FUNNEL_FLAGS.items():
        for p in patterns:
            if re.search(p, query_l):
                flags.append(key)
                break
    return {"devices": devices, "categories": categories, "flags": flags}


def _connect_neo4j():
    uri = get_env("NEO4J_URI")
    user = get_env("NEO4J_USER")
    password = get_env("NEO4J_PASSWORD")
    if uri and user and password:
        return GraphDatabase.driver(uri, auth=basic_auth(user, password))
    return None


def _cypher_filter_where(filters: Dict[str, List[str]]) -> Tuple[str, Dict[str, object]]:
    where_clauses = []
    params: Dict[str, object] = {}

    # We only have summary_text; filter by CONTAINS on keywords
    if filters.get("devices"):
        dev_clauses = []
        for i, dev in enumerate(filters["devices"]):
            key = f"dev{i}"
            dev_clauses.append(f"toLower(c.summary_text) CONTAINS ${key}")
            params[key] = dev.lower()
        where_clauses.append("(" + " OR ".join(dev_clauses) + ")")

    if filters.get("categories"):
        cat_clauses = []
        for i, cat in enumerate(filters["categories"]):
            key = f"cat{i}"
            cat_clauses.append(f"toLower(c.summary_text) CONTAINS ${key}")
            params[key] = cat.lower()
        where_clauses.append("(" + " OR ".join(cat_clauses) + ")")

    if filters.get("flags"):
        flag_clauses = []
        for i, flag in enumerate(filters["flags"]):
            key = f"flag{i}"
            flag_clauses.append(f"toLower(c.summary_text) CONTAINS ${key}")
            params[key] = flag.lower()
        where_clauses.append("(" + " OR ".join(flag_clauses) + ")")

    where_str = " AND ".join(where_clauses) if where_clauses else ""
    return where_str, params


def fetch_candidate_clusters(filters: Dict[str, List[str]]) -> List[int]:
    driver = _connect_neo4j()
    if not driver:
        # Fallback: apply lightweight filtering over local parquet summary_text
        df = pd.read_parquet(data_dir("processed", "clusters.parquet"))
        if not filters.get("devices") and not filters.get("categories") and not filters.get("flags"):
            return df["cluster_id"].tolist()

        def row_matches(summary: str) -> bool:
            s = str(summary).lower()
            clauses = []
            if filters.get("devices"):
                clauses.append(any(dev.lower() in s for dev in filters["devices"]))
            if filters.get("categories"):
                clauses.append(any(cat.lower() in s for cat in filters["categories"]))
            if filters.get("flags"):
                clauses.append(any(flag.lower() in s for flag in filters["flags"]))
            return all(clauses) if clauses else True

        mask = df["summary_text"].apply(row_matches)
        return df.loc[mask, "cluster_id"].tolist()

    where_str, params = _cypher_filter_where(filters)
    cypher = "MATCH (c:Cluster)"
    if where_str:
        cypher += f" WHERE {where_str}"
    cypher += " RETURN c.cluster_id AS cluster_id"

    with driver.session() as session:
        result = session.run(cypher, parameters=params)
        cluster_ids = [int(r["cluster_id"]) for r in result]
    driver.close()
    return cluster_ids


def retrieve_clusters(query: str, top_n: int = 10) -> List[dict]:
    # Parse filters
    filters = parse_filters(query)

    # Graph-side candidates
    candidate_ids = set(fetch_candidate_clusters(filters))

    # Vector search
    model = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)

    vec_index = ClusterVectorIndex.load()
    distances, indices = vec_index.search(query_vec, top_k=min(top_n * 5, len(vec_index.ids)))

    # Combine scores
    clusters_df = pd.read_parquet(data_dir("processed", "clusters.parquet"))
    ranked: List[tuple[int, float]] = []
    for score, idx in zip(distances[0].tolist(), indices[0].tolist()):
        cluster_id = int(clusters_df.iloc[idx]["cluster_id"])  # index order matches embeddings
        # Filter bonus if in graph candidates
        filter_bonus = 0.2 if (not filters["devices"] and not filters["categories"] and not filters["flags"]) else (0.2 if cluster_id in candidate_ids else 0.0)
        combined = float(score) + filter_bonus
        ranked.append((cluster_id, combined))

    # Deduplicate and take top_n
    seen = set()
    ranked_unique: List[tuple[int, float]] = []
    for cid, sc in sorted(ranked, key=lambda x: x[1], reverse=True):
        if cid not in seen:
            ranked_unique.append((cid, sc))
            seen.add(cid)
        if len(ranked_unique) >= top_n:
            break

    # Build output
    out = []
    cdf = clusters_df.set_index("cluster_id")
    for cid, sc in ranked_unique:
        row = cdf.loc[cid]
        out.append(
            {
                "cluster_id": int(cid),
                "score": float(sc),
                "size": int(row["size"]),
                "summary_text": str(row["summary_text"]),
                "sample_muids": [int(x) for x in list(row["sample_muids"])],
            }
        )
    return out


def main() -> None:
    results = retrieve_clusters("android users who abandoned checkout", top_n=5)
    for r in results:
        print(r)


if __name__ == "__main__":
    main()


