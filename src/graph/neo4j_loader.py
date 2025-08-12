from __future__ import annotations

import pandas as pd
from neo4j import GraphDatabase

from ..common.utils import data_dir, get_env


def _get_driver():
    uri = get_env("NEO4J_URI")
    user = get_env("NEO4J_USER")
    password = get_env("NEO4J_PASSWORD")
    if not uri or not user or not password:
        raise RuntimeError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set in environment")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver


def load_clusters_to_neo4j(store_embeddings: bool = False) -> None:
    clusters_path = data_dir("processed", "clusters.parquet")
    df = pd.read_parquet(clusters_path)

    driver = _get_driver()
    with driver.session() as session:
        # Constraints
        session.run("CREATE CONSTRAINT cluster_id_unique IF NOT EXISTS FOR (c:Cluster) REQUIRE c.cluster_id IS UNIQUE")
        session.run("CREATE CONSTRAINT customer_id_unique IF NOT EXISTS FOR (c:Customer) REQUIRE c.id IS UNIQUE")

        # Upsert clusters
        for row in df.itertuples(index=False):
            params = {
                "cluster_id": int(row.cluster_id),
                "size": int(row.size),
                "summary_text": str(row.summary_text),
                "embedding": getattr(row, "embedding", None) if store_embeddings else None,
                "sample_muids": list(row.sample_muids),
            }
            session.run(
                "MERGE (c:Cluster {cluster_id: $cluster_id}) "
                "SET c.size = $size, c.summary_text = $summary_text "
                + (", c.embedding = $embedding" if store_embeddings else "")
                , parameters=params
            )
            # Sampled customers and relationships
            for muid in params["sample_muids"]:
                session.run(
                    "MERGE (u:Customer {id: $id}) "
                    "WITH u MATCH (c:Cluster {cluster_id: $cluster_id}) "
                    "MERGE (u)-[:IN_CLUSTER]->(c)",
                    parameters={"id": int(muid), "cluster_id": int(row.cluster_id)},
                )

    driver.close()


def export_graph_for_vis(limit_clusters: int = 200) -> str:
    raise NotImplementedError("Export visualization has been removed from the minimal deliverable.")

def main() -> None:
    load_clusters_to_neo4j(store_embeddings=False)
    print("Loaded clusters into Neo4j")


if __name__ == "__main__":
    main()


