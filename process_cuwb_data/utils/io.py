import datetime
import os
import pickle

import pandas as pd

from .log import logger


def write_datafile_to_csv(df, filename, directory=".", index=True):
    filename = filename + ".csv"
    path = os.path.join(directory, filename)
    logger.info(f"Writing datafile '{filename}' to {path}")
    df.to_csv(path, index=index)


def write_generic_pkl(record, filename, directory="."):
    filename = filename + ".pkl"
    path = os.path.join(directory, filename)
    with open(path, "wb") as fp:
        logger.info(f"Writing pickle '{filename}' record to {path}")
        pickle.dump(record, fp)


def read_generic_pkl(path):
    with open(path, "rb") as fp:
        record = pickle.load(fp)
        logger.info(f"Loaded pickle record '{path}', type '{type(record).__name__}'")

    return record


def write_cuwb_data_pkl(
    df,
    filename_prefix,
    environment_name,
    start_time,
    end_time,
    entity_type=None,
    data_type=None,
    environment_assignment_info=False,
    entity_assignment_info=False,
    include_meta_fields=False,
    fillna=None,
    directory=".",
):
    os.makedirs(directory, exist_ok=True)

    path = cuwb_data_path(
        filename_prefix=filename_prefix,
        environment_name=environment_name,
        start_time=start_time,
        end_time=end_time,
        entity_type=entity_type,
        data_type=data_type,
        environment_assignment_info=environment_assignment_info,
        entity_assignment_info=entity_assignment_info,
        include_meta_fields=include_meta_fields,
        fillna=fillna,
        directory=directory,
    )
    if df is None:
        logger.warning(f"Cannot write CUWB data to pickle, dataframe provided == None: {path}")
        return

    logger.info(f"Writing CUWB data to {path}")
    df.to_pickle(path)


def read_cuwb_data_pkl(
    filename_prefix,
    environment_name,
    start_time,
    end_time,
    entity_type=None,
    data_type=None,
    environment_assignment_info=False,
    entity_assignment_info=False,
    include_meta_fields=False,
    fillna=None,
    directory=".",
):
    path = cuwb_data_path(
        filename_prefix=filename_prefix,
        environment_name=environment_name,
        start_time=start_time,
        end_time=end_time,
        entity_type=entity_type,
        data_type=data_type,
        environment_assignment_info=environment_assignment_info,
        entity_assignment_info=entity_assignment_info,
        include_meta_fields=include_meta_fields,
        fillna=fillna,
        directory=directory,
    )

    if not os.path.exists(path):
        logger.info(f"Unable to read CUWB data from {path}, path does not exist")
        return None

    logger.info(f"Reading CUWB data from {path}")
    df = pd.read_pickle(path)
    return df


def cuwb_data_path(
    filename_prefix,
    environment_name,
    start_time,
    end_time,
    entity_type=None,
    data_type=None,
    environment_assignment_info=False,
    entity_assignment_info=False,
    include_meta_fields=False,
    fillna=None,
    directory=".",
):
    start_time_string = "None"
    if start_time is not None:
        start_time_string = datetime_filename_format(start_time)
    end_time_string = "None"
    if end_time is not None:
        end_time_string = datetime_filename_format(end_time)
    filename = "-".join([filename_prefix, environment_name, start_time_string, end_time_string])
    if entity_type:
        filename = f"{filename}-(entity_type_{entity_type})"
    if data_type:
        filename = f"{filename}-(data_type_{data_type})"
    if environment_assignment_info:
        filename = f"{filename}-(env_assignments)"
    if entity_assignment_info:
        filename = f"{filename}-(entity_assignments)"
    if include_meta_fields:
        filename = f"{filename}-(include_meta_fields)"
    if fillna:
        filename = f"{filename}-(fillna-{fillna})"
    filename = filename + ".pkl"
    path = os.path.join(directory, filename)
    return path


def datetime_filename_format(timestamp):
    return timestamp.astimezone(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")


def load_csv(path):
    df = pd.read_csv(path)

    df.rename(columns={"start_time": "start_datetime", "end_time": "end_datetime"}, inplace=True)

    if "start_datetime" in df.columns:
        df["start_datetime"] = pd.to_datetime(df["start_datetime"], format="ISO8601")
    if "end_datetime" in df.columns:
        df["end_datetime"] = pd.to_datetime(df["end_datetime"], format="ISO8601")

    if len(df) == 0:
        return df

    # Recognize a supplied timezone but if missing assume UTC time was implied
    # TODO: handle per row timezone differences
    if (
        df["start_datetime"][0].tzinfo is None
        or df["start_datetime"][0].tzinfo.utcoffset(df["start_datetime"][0]) is None
    ):
        df["start_datetime"] = df["start_datetime"].dt.tz_localize("UTC")
        df["end_datetime"] = df["end_datetime"].dt.tz_localize("UTC")
    else:
        df["start_datetime"] = df["start_datetime"].dt.tz_convert("UTC")
        df["end_datetime"] = df["end_datetime"].dt.tz_convert("UTC")

    return df
