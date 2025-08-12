from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..common.utils import data_dir, write_json, read_json


try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False


@dataclass
class ClusterVectorIndex:
    index: object
    ids: List[int]
    use_faiss: bool

    def search(self, query_vectors: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        if self.use_faiss:
            distances, indices = self.index.search(query_vectors.astype(np.float32), top_k)
            return distances, indices
        # NumPy fallback using cosine similarity (vectors are normalized in embeddings.py and graphrag)
        embeddings = self.index  # type: ignore[assignment]
        sims = query_vectors @ embeddings.T  # (q, d)
        # argsort descending
        idxs = np.argsort(-sims, axis=1)[:, :top_k]
        scores = np.take_along_axis(sims, idxs, axis=1)
        return scores.astype(np.float32), idxs.astype(np.int64)

    def save(self, index_path: Path | None = None, ids_path: Path | None = None) -> None:
        index_path = index_path or data_dir("processed", "faiss.index")
        ids_path = ids_path or data_dir("processed", "faiss_ids.json")
        if self.use_faiss:
            faiss.write_index(self.index, str(index_path))  # type: ignore[arg-type]
        write_json(ids_path, self.ids)

    @staticmethod
    def load(index_path: Path | None = None, ids_path: Path | None = None) -> "ClusterVectorIndex":
        index_path = index_path or data_dir("processed", "faiss.index")
        ids_path = ids_path or data_dir("processed", "faiss_ids.json")
        ids = read_json(ids_path)
        if _FAISS_AVAILABLE and Path(index_path).exists():
            index = faiss.read_index(str(index_path))  # type: ignore[assignment]
            return ClusterVectorIndex(index=index, ids=ids, use_faiss=True)
        # Fallback: use embeddings directly
        embeddings = np.load(data_dir("processed", "cluster_embeddings.npy")).astype(np.float32)
        return ClusterVectorIndex(index=embeddings, ids=ids, use_faiss=False)


def build_and_save_index() -> ClusterVectorIndex:
    clusters_path = data_dir("processed", "clusters.parquet")
    emb_path = data_dir("processed", "cluster_embeddings.npy")

    clusters_df = pd.read_parquet(clusters_path)
    embeddings = np.load(emb_path).astype(np.float32)

    if _FAISS_AVAILABLE:
        # Using inner product with normalized embeddings equals cosine similarity
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # type: ignore[attr-defined]
        index.add(embeddings)
        vec_index = ClusterVectorIndex(index=index, ids=clusters_df["cluster_id"].tolist(), use_faiss=True)
        vec_index.save()
        return vec_index
    # Fallback: use numpy-backed index; still save ids for load()
    vec_index = ClusterVectorIndex(index=embeddings, ids=clusters_df["cluster_id"].tolist(), use_faiss=False)
    write_json(data_dir("processed", "faiss_ids.json"), vec_index.ids)
    return vec_index


def main() -> None:
    build_and_save_index()
    print("Saved FAISS index to", data_dir("processed", "faiss.index"))


if __name__ == "__main__":
    main()


