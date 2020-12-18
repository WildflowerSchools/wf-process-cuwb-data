def dataframe_tuple_columns_to_underscores(df, inplace=False):
    if not inplace:
        df = df.copy()

    def rename(col):
        if isinstance(col, tuple):
            col = list(filter(None, col))  # Remove empty strings from col names
            col = '_'.join(str(c) for c in col)
        return col

    df.columns = map(rename, df.columns)

    if not inplace:
        return df
