import datetime
import hashlib
import json
from typing import Union


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


def filter_by_entity_type(df, entity_type="all"):
    if entity_type.lower() == "all":
        return df
    if entity_type.lower() == "tray":
        return df.loc[df["entity_type"].str.lower() == "tray"].copy()
    if entity_type.lower() == "person":
        return df.loc[df["entity_type"].str.lower() == "person"].copy()

    error = f"Invalid 'entity_type' value: {entity_type}"
    raise ValueError(error)


def filter_by_data_type(df, data_type="all"):
    if data_type == "all" or data_type is None or len(df) == 0:
        return df

    if data_type not in ["position", "accelerometer", "gyroscope", "magnetometer"]:
        error = f"Invalid 'data_type' value: {data_type}"
        raise ValueError(error)

    return df.loc[df["type"] == data_type].copy()


def map_column_name_to_dimension_space(column_name, num_dimensions):
    dims = ["x", "y", "z"]
    return list(map(lambda d: f"{d}_{column_name}", dims[0:num_dimensions]))


def checksum(item: Union[dict, str, bytes]):
    byte_string = None
    if isinstance(item, bytes):
        byte_string = item
    elif isinstance(item, dict):

        def json_serial(obj):
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()

            raise TypeError(f"Type {type(obj)} not serializable")

        byte_string = json.dumps(item, default=json_serial).encode()
    elif isinstance(item, str):
        byte_string = bytes(item, "utf-8")
    if byte_string is None:
        raise ValueError("Unexpected item passed to checksum(), expected Union[dict, str, bytes]")

    return hashlib.sha256(byte_string).hexdigest()
