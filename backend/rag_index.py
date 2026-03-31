import json
import os
from typing import List, Tuple, Any

import faiss
import numpy as np


def load_index(index_path: str, meta_path: str) -> Tuple[faiss.IndexFlatL2, np.ndarray]:
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise RuntimeError(
            "FAISS index or metadata not found. "
            "Run `python backend/build_index.py` first to build the RAG index."
        )

    index = faiss.read_index(index_path)
    metadata = np.load(meta_path, allow_pickle=True)
    return index, metadata


def search_similar(
    index: faiss.IndexFlatL2,
    metadata: np.ndarray,
    query_vec: np.ndarray,
    k: int = 3,
) -> Tuple[List[Any], List[float]]:
    """Возвращает (результаты, дистанции) — дистанции нужны для метрик."""
    distances, indices = index.search(query_vec, k)
    idxs = indices[0]
    dists = distances[0]
    results = []
    scores = []
    for i, d in zip(idxs, dists):
        if 0 <= i < len(metadata):
            results.append(metadata[i])
            scores.append(float(round(d, 4)))
    return results, scores


def load_faq_data(path: str):
    """Загружает FAQ данные из JSON файла."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


