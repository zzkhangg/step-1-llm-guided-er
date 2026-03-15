from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def prune_attributes(df, miss_thres=0.5, unique_thres=0.01):
    missing_ratio = df.isnull().mean()
    unique_ratio = df.nunique() / len(df)
    keep = [
        col for col in df.columns
        if missing_ratio[col] < miss_thres
        and unique_ratio[col] > unique_thres
    ]
    return df[keep]


def build_tfidf_compressor(df_A, df_B, cols, top_n=10, min_tokens=5):
    
    # pre-compute avg token length per column across both tables
    col_avg_tokens = {}
    for col in cols:
        vals_A = df_A[col] if col in df_A.columns else pd.Series()
        vals_B = df_B[col] if col in df_B.columns else pd.Series()
        vals   = pd.concat([vals_A, vals_B]).dropna().astype(str)
        col_avg_tokens[col] = vals.str.split().apply(len).mean()
        print(f"[{col}] avg_tokens={col_avg_tokens[col]:.1f} → "
              f"{'compress' if col_avg_tokens[col] > min_tokens else 'keep as-is'}")

    # fit one TF-IDF per column
    tfidf_per_col = {}
    for col in cols:
        if col_avg_tokens[col] <= min_tokens:
            continue  # skip fitting TF-IDF for short columns — not needed
        vals_A = df_A[col] if col in df_A.columns else pd.Series()
        vals_B = df_B[col] if col in df_B.columns else pd.Series()
        corpus = pd.concat([vals_A, vals_B]).fillna('').astype(str).str.lower().tolist()
        tfidf  = TfidfVectorizer(token_pattern=r'\b\w{2,}\b', sublinear_tf=True,
                                 min_df=2, max_df=0.95)
        tfidf.fit(corpus)
        vocab = tfidf.get_feature_names_out()
        tfidf_per_col[col] = dict(zip(vocab, tfidf.idf_))

    def compress_attribute(value, col):
        if pd.isna(value) or str(value).strip() == '':
            return ''

        # skip compression if column avg is short — keep entire value as-is
        if col_avg_tokens.get(col, 0) <= min_tokens:
            return str(value).lower().strip()

        tokens = str(value).lower().split()
        idf    = tfidf_per_col.get(col, {})
        scored = [(token, idf.get(token, 0.0)) for token in tokens]

        seen, kept = set(), []
        for token, score in sorted(scored, key=lambda x: -x[1]):
            if token not in seen:
                seen.add(token)
                kept.append(token)
            if len(kept) >= top_n:
                break
        return ' '.join(kept)

    return compress_attribute


def heuristic_selection(df_A, df_B):
    df_A = prune_attributes(df_A)
    df_B = prune_attributes(df_B)
    cols = df_A.columns.tolist()

    compress_attribute = build_tfidf_compressor(df_A, df_B, cols, top_n=10, min_tokens=5)
    for col in cols:
        df_A[col] = df_A[col].apply(lambda val: compress_attribute(val, col))
        df_B[col] = df_B[col].apply(lambda val: compress_attribute(val, col))
    return df_A, df_B