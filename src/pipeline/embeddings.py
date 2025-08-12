from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from ..common.utils import data_dir


MODEL_NAME = "all-MiniLM-L6-v2"


def embed_cluster_summaries(batch_size: int = 64) -> pd.DataFrame:
    clusters_path = data_dir("processed", "clusters.parquet")
    df = pd.read_parquet(clusters_path)

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(df["summary_text"].tolist(), batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Save .npy
    emb_path = data_dir("processed", "cluster_embeddings.npy")
    np.save(emb_path, embeddings)

    # Also store in metadata (as lists)
    df["embedding"] = [emb.astype(float).tolist() for emb in embeddings]
    df.to_parquet(clusters_path, index=False)

    return df


def main() -> None:
    embed_cluster_summaries()
    print("Saved cluster embeddings to", data_dir("processed", "cluster_embeddings.npy"))


if __name__ == "__main__":
    main()


