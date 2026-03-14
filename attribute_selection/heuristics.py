def prune_attributes(df, miss_thres=0.5, unique_thres=0.01):
    missing_ratio = df.isnull().mean()
    unique_ratio = df.nunique() / len(df)
    keep = [
        col for col in df.columns
        if missing_ratio[col] < miss_thres
        and unique_ratio[col] > unique_thres
    ]
    return df[keep]


