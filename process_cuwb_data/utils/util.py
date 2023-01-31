def dataframe_tuple_columns_to_underscores(df, inplace=False):
    if not inplace:
        df = df.copy()

    def rename(col):
        if isinstance(col, tuple):
            col = list(filter(None, col))  # Remove empty strings from col names
            col = "_".join(str(c) for c in col)
        return col

    df.columns = map(rename, df.columns)

    if not inplace:
        return df


def filter_entity_type(dataframe, entity_type="all"):
    if entity_type.lower() == "all":
        return dataframe
    if entity_type.lower() == "tray":
        return dataframe.loc[dataframe["entity_type"].str.lower() == "tray"].copy()
    if entity_type.lower() == "person":
        return dataframe.loc[dataframe["entity_type"].str.lower() == "person"].copy()

    error = f"Invalid 'entity_type' value: {entity_type}"
    raise ValueError(error)


def filter_data_type_and_format(df, data_type="all"):
    if data_type == "all" or data_type is None or len(df) == 0:
        return df

    if data_type not in ["position", "accelerometer", "gyroscope", "magnetometer"]:
        error = f"Invalid 'data_type' value: {data_type}"
        raise ValueError(error)

    return df.loc[df["type"] == data_type].copy()
