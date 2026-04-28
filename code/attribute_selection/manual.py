def manual_selection(df_A, df_B, MARKERS_ATTRIBUTES):
    df_A = df_A[MARKERS_ATTRIBUTES]
    df_B = df_B[MARKERS_ATTRIBUTES]
    return df_A, df_B