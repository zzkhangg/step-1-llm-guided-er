import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Where per-column embeddings are memoised. Encoding a table is deterministic given the
# model and the exact column contents, but it is not cheap: on Amazon-Walmart it is five
# passes over 22,074 records, and a run that dies before the matcher has spent that time
# for nothing. The cache makes a restart resume in seconds, which matters because the
# runs that die are exactly the long ones.
EMBED_CACHE_DIR = os.getenv("EMBED_CACHE_DIR", "cache/embeddings")


def set_embed_cache_dir(path):
    global EMBED_CACHE_DIR
    EMBED_CACHE_DIR = str(path)


def _column_cache_key(model_name, model, col, texts):
    """Identify a column encoding by model, column name, and exact contents."""
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8", "replace"))
    h.update(str(model.get_sentence_embedding_dimension()).encode())
    h.update(b"\x00" + col.encode("utf-8", "replace") + b"\x00")
    for t in texts:
        h.update(t.encode("utf-8", "replace"))
        h.update(b"\x1f")
    return h.hexdigest()


def _load_cached(key):
    if not EMBED_CACHE_DIR:
        return None
    f = Path(EMBED_CACHE_DIR) / f"{key}.npy"
    if f.exists():
        try:
            return np.load(f)
        except Exception:
            # A truncated file from an interrupted write is not worth diagnosing;
            # re-encoding is always correct.
            return None
    return None


def _save_cached(key, arr):
    if not EMBED_CACHE_DIR:
        return
    d = Path(EMBED_CACHE_DIR)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{key}.tmp{os.getpid()}.npy"
    # Write then rename: a kill mid-write must not leave a half file that later loads
    # as valid-looking garbage. The handle is opened here rather than passed as a path
    # because np.save appends ".npy" to any path lacking it, which would rename the
    # temporary out from under the replace below.
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, d / f"{key}.npy")


def embed_dataframe_sbert(df, model, batch_size=256, model_name=None):
    """
    Batch encode all values per column,
    then concatenate column embeddings per record.

    ``model_name`` identifies the encoder in the cache key. Caching is skipped when it is
    not supplied: an embedding served under a key that does not pin the model would
    silently return another encoder's vectors, which is a worse failure than re-encoding.
    """
    dim  = model.get_sentence_embedding_dimension()
    cols = df.columns.tolist()
    use_cache = bool(EMBED_CACHE_DIR) and bool(model_name)

    col_embeddings = {}
    n_cached = 0
    for col in cols:
        texts = [
            str(v).strip() if not pd.isna(v) and str(v).strip() != '' else ''
            for v in df[col]
        ]
        key = _column_cache_key(model_name, model, col, texts) if use_cache else None
        embeddings = _load_cached(key) if use_cache else None
        if embeddings is not None and embeddings.shape[0] == len(texts):
            n_cached += 1
        else:
            # batch encode entire column at once
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False
            )
            # zero out empty values
            for i, t in enumerate(texts):
                if t == '':
                    embeddings[i] = np.zeros(dim)
            if use_cache:
                _save_cached(key, embeddings)
        col_embeddings[col] = embeddings

    if n_cached:
        print(f"  [embed] {n_cached}/{len(cols)} columns loaded from cache")

    # concatenate column vectors per record
    return normalize(np.hstack([col_embeddings[col] for col in cols]))