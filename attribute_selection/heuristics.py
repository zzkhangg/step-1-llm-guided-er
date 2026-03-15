import pandas as pd
import re

def prune_attributes(df_A, df_B, miss_thresh=0.5, unique_thresh=0.01):
    cols = [c for c in df_A.columns if c in df_B.columns]
    keep = []
    for col in cols:
        vals_A = df_A[col] if col in df_A.columns else pd.Series()
        vals_B = df_B[col] if col in df_B.columns else pd.Series()
        combined = pd.concat([vals_A, vals_B])
        missing_ratio = combined.isnull().mean()

        vals = combined.dropna()
        n_unique = vals.nunique()
        unique_ratio = n_unique / len(vals) if len(vals) > 0 else 0

        # keep if: low missingness AND (reasonable unique ratio OR at least 2 unique values)
        if missing_ratio < miss_thresh and unique_ratio > unique_thresh:
            keep.append(col)
        else:
            print(f"[{col}] PRUNED — missing={missing_ratio:.3f}, "
                  f"unique_ratio={unique_ratio:.4f}, n_unique={n_unique}")

    print(f"Kept cols: {keep}")
    return df_A[keep].copy(), df_B[keep].copy()

# ── high-signal patterns to KEEP ──
HIGH_SIGNAL_PATTERNS = [
    r'\b\d{2,}\b',                                     # Numbers
    r'\b[A-Z0-9]+(?:[-_][A-Z0-9]+)*\b',               # Codes/IDs
    r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b',            # Proper names / capitalized words
    r'\b\d+\.?\d*\s*(?:kg|g|lb|lbs|oz|cm|mm|m|km|in|inch|inches|ft|l|ml|v|w|hz|ghz|mp)\b',  # Units
    r'\b[a-z]{2,}\b',                                 # Generic lowercase words
]

BOILERPLATE_PATTERNS = [
    r'http\S+',
    r'www\.\S+',
    r'\b(the|a|an|and|or|of|for|in|on|at|to|with|by|from|is|it|this|that|as)\b',
    r'[^\w\s]',  # punctuation
    r'\b(?:free|shipping|included|certified|refurbished|brand|item|product|please|note|click|here|more|details|information|features|description)\b'
]


high_signal_regex = [re.compile(p, re.IGNORECASE) for p in HIGH_SIGNAL_PATTERNS]
boilerplate_regex = [re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS]

def compress_text(text: str, min_tokens: int = 10) -> str:
    if not isinstance(text, str):
        return ""
    
    tokens = text.strip().split()
    
    # skip compression for short values — already concise
    if len(tokens) <= min_tokens:
        return text.lower().strip()

    # remove boilerplate
    cleaned = text
    for pattern in boilerplate_regex:
        cleaned = pattern.sub(" ", cleaned)

    # extract high-signal tokens
    seen, result = set(), []
    for token in cleaned.split():
        t_lower = token.lower().strip()
        if not t_lower:
            continue
        if any(p.fullmatch(token) for p in high_signal_regex):
            if t_lower not in seen:
                seen.add(t_lower)
                result.append(token)

    return " ".join(result) if result else cleaned.strip()

def compress_dataframe(df):
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].apply(compress_text)
    return df
def heuristic_selection(df_A, df_B):
    df_A, df_B = prune_attributes(df_A, df_B)
    df_A = compress_dataframe(df_A)
    df_B = compress_dataframe(df_B)

    return df_A, df_B