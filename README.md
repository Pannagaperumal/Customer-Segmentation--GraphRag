## Customer Journey Graph + GraphRAG (Python)

This project builds a lightweight Customer Journey Graph and a GraphRAG retrieval system from synthetic Users and Events data. Raw events are stored outside Neo4j (Parquet). Neo4j stores only clusters and up to 10 sampled customers per cluster.

### What it does
- Generates synthetic `Users` and `Events` data
- Aggregates per-user features and saves to Parquet
- Clusters users (KMeans) into 50 clusters and summarizes each cluster
- Embeds cluster summaries (Sentence-Transformers `all-MiniLM-L6-v2`)
- Builds a FAISS vector index for fast semantic retrieval
- Loads clusters to Neo4j (Cluster and sampled Customer nodes only)
- Provides a GraphRAG retriever that combines:
  - graph-side filtering via Cypher (text search over `summary_text`)
  - semantic similarity over cluster summary embeddings
- Exposes `/ask` FastAPI endpoint and a Typer CLI

### Project layout
- `src/pipeline/data_prep.py`: Generate/load data and create `user_features.parquet`
- `src/pipeline/clustering.py`: Scale, cluster, summarize, and save `clusters.parquet`
- `src/pipeline/embeddings.py`: Embed `summary_text` and save `.npy` + Parquet
- `src/retrieval/vector_index.py`: Build and persist FAISS index
- `src/graph/neo4j_loader.py`: Load clusters + sampled customers into Neo4j
- `src/retrieval/graphrag.py`: GraphRAG retrieval combining Neo4j filter + FAISS
- `src/api/ask_api.py`: FastAPI server with `/ask` and raw data endpoints
- `src/cli.py`: Typer CLI for local use
- `examples/`: Example queries and sample outputs

### Quickstart
1) Install

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2) Generate data and build artifacts

```bash
python -m src.cli prep
python -m src.cli cluster
python -m src.cli embed
python -m src.cli index
```

3) (Optional) Load into Neo4j

Set environment variables:

```bash
setx NEO4J_URI bolt+s://<your-auradb-uri>:7687
setx NEO4J_USER neo4j
setx NEO4J_PASSWORD <password>
```

Then:

```bash
python -m src.cli load-neo4j
```

4) Ask via CLI

```bash
python -m src.cli ask "show clusters of android users who added to cart but not purchased"
```

5) Ask via API

```bash
uvicorn src.api.ask_api:app --host 0.0.0.0 --port 8000
# Then POST {"query": "...", "top_n": 5} to http://localhost:8000/ask
```

### Data outputs
- `data/processed/user_features.parquet`
- `data/processed/clusters.parquet`
- `data/processed/cluster_embeddings.npy`
- `data/processed/faiss.index`

Raw synthetic inputs are also saved to `data/raw/users.parquet` and `data/raw/events.parquet`.

### Notes
- Only `Cluster` and sampled `Customer` nodes are written to Neo4j. Raw events remain in Parquet.
- Graph filtering uses `summary_text` in Neo4j for lightweight matching (e.g., device OS mentions, funnel flags). If Neo4j is unavailable, retrieval falls back to local Parquet + FAISS.
