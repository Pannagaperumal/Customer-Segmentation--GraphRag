from __future__ import annotations

import json

import typer

from .common.utils import data_dir


app = typer.Typer(help="Customer Journey Graph + GraphRAG pipeline")


@app.command()
def prep():
    """Generate synthetic data and build user features."""
    from .pipeline.data_prep import generate_synthetic_data, build_user_features

    users_df, events_df = generate_synthetic_data()
    build_user_features(users_df, events_df)
    typer.echo(f"Saved: {data_dir('processed', 'user_features.parquet')}")


@app.command()
def cluster(k: int = 50):
    """Cluster users and save clusters."""
    # Lazy import to avoid importing heavy deps from other commands (e.g., torch)
    from .pipeline.clustering import cluster_users

    cluster_users(k=k)
    typer.echo(f"Saved: {data_dir('processed', 'clusters.parquet')}")


@app.command()
def embed():
    """Embed cluster summaries and save embeddings."""
    from .pipeline.embeddings import embed_cluster_summaries

    embed_cluster_summaries()
    typer.echo(f"Saved: {data_dir('processed', 'cluster_embeddings.npy')}")


@app.command()
def index():
    """Build and save FAISS index."""
    from .retrieval.vector_index import build_and_save_index

    build_and_save_index()
    typer.echo(f"Saved: {data_dir('processed', 'faiss.index')}")


@app.command("build-all")
def build_all(k: int = 50):
    """Run the full pipeline: prep -> cluster -> embed -> index."""
    # Lazy imports to avoid importing heavy deps unless needed
    from .pipeline.data_prep import generate_synthetic_data, build_user_features
    from .pipeline.clustering import cluster_users
    from .pipeline.embeddings import embed_cluster_summaries
    from .retrieval.vector_index import build_and_save_index

    # Prep
    users_df, events_df = generate_synthetic_data()
    build_user_features(users_df, events_df)
    typer.echo(f"Saved: {data_dir('processed', 'user_features.parquet')}")

    # Cluster
    cluster_users(k=k)
    typer.echo(f"Saved: {data_dir('processed', 'clusters.parquet')}")

    # Embed
    embed_cluster_summaries()
    typer.echo(f"Saved: {data_dir('processed', 'cluster_embeddings.npy')}")

    # Index
    build_and_save_index()
    typer.echo(f"Saved: {data_dir('processed', 'faiss.index')}")


@app.command("load-neo4j")
def load_neo4j(store_embeddings: bool = False):
    """Load clusters and sampled customers into Neo4j."""
    from .graph.neo4j_loader import load_clusters_to_neo4j

    load_clusters_to_neo4j(store_embeddings=store_embeddings)
    typer.echo("Loaded clusters into Neo4j")


@app.command("export-graph")
def export_graph(limit_clusters: int = 200):
    """Export a small Neo4j subgraph to examples/graph.html for visualization."""
    typer.echo("This command has been removed.")


@app.command()
def ask(query: str, top_n: int = 5):
    """Run GraphRAG retrieval from CLI."""
    from .retrieval.graphrag import retrieve_clusters

    results = retrieve_clusters(query, top_n=top_n)
    typer.echo(json.dumps({"query": query, "results": results}, indent=2))


# Raw data inspection commands
raw_app = typer.Typer(help="Inspect raw Parquet data (users/events)")
app.add_typer(raw_app, name="raw")


@raw_app.command("users")
def raw_users(limit: int = typer.Option(20, "--limit", min=1, help="Max rows to show")):
    """Show first N rows from data/raw/users.parquet."""
    import pandas as pd

    users_path = data_dir("raw", "users.parquet")
    if not users_path.exists():
        raise typer.Exit("Raw users parquet not found. Run: python -m src.cli prep")
    df = pd.read_parquet(users_path).head(limit)
    typer.echo(json.dumps(df.to_dict(orient="records"), indent=2, default=str))


@raw_app.command("events")
def raw_events(limit: int = typer.Option(20, "--limit", min=1, help="Max rows to show")):
    """Show first N rows from data/raw/events.parquet."""
    import pandas as pd

    events_path = data_dir("raw", "events.parquet")
    if not events_path.exists():
        raise typer.Exit("Raw events parquet not found. Run: python -m src.cli prep")
    df = pd.read_parquet(events_path).head(limit)
    typer.echo(json.dumps(df.to_dict(orient="records"), indent=2, default=str))


if __name__ == "__main__":
    app()


