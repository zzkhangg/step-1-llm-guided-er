import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

def embed_dataframe_sbert(df, model, batch_size=256):
    """
    Batch encode all values per column,
    then concatenate column embeddings per record.
    """
    dim  = model.get_sentence_embedding_dimension()
    cols = df.columns.tolist()

    col_embeddings = {}
    for col in cols:
        texts = [
            str(v).strip() if not pd.isna(v) and str(v).strip() != '' else ''
            for v in df[col]
        ]
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
        col_embeddings[col] = embeddings

    # concatenate column vectors per record
    return normalize(np.hstack([col_embeddings[col] for col in cols]))