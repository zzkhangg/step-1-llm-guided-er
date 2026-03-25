import numpy as np
import pandas as pd
import re
from .constants import *


def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    return tokens

def embed_text(tokens, embedding_model):
    vecs = []
    for token in tokens:
        if token in embedding_model:
            vecs.append(embedding_model[token])

    if len(vecs) == 0:
        return np.zeros(embedding_model.vector_size)

    return np.mean(vecs, axis=0)

def record_to_vector(row, embedding_model):
    vecs = []
    for val in row.values:
        try:
            is_empty = pd.isna(val) or str(val).strip() == ''
        except:
            is_empty = False

        if is_empty:
            vec = np.zeros(embedding_model.vector_size)
        else:
            vec = embed_text(preprocess(str(val)), embedding_model)
        vecs.append(vec)

    # use len(row) instead of hardcoded MARKERS_ATTRIBUTES
    return np.concatenate(vecs) if vecs else np.zeros(embedding_model.vector_size * len(row))