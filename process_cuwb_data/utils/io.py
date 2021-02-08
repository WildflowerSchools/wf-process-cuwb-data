import datetime
import os
import pickle

import pandas as pd

from .log import logger


def write_datafile_to_csv(df, filename, directory='.', index=True):
    filename = filename + '.csv'
    path = os.path.join(
        directory,
        filename
    )
    logger.info("Writing datafile '{}' to {}".format(filename, path))
    df.to_csv(path, index=index)


def write_generic_pkl(record, filename, directory='.'):
    filename = filename + '.pkl'
    path = os.path.join(
        directory,
        filename
    )
    with open(path, 'wb') as fp:
        logger.info("Writing pickle '{}' record to {}".format(filename, path))
        pickle.dump(record, fp)


def read_generic_pkl(path):
    with open(path, 'rb') as fp:
        record = pickle.load(fp)
        logger.info("Loaded pickle record '{}', type '{}'".format(path, type(record).__name__))

    return record


def write_cuwb_data_pkl(
        df,
        filename_prefix,
        environment_name,
        start_time,
        end_time,
        environment_assignment_info=False,
        entity_assignment_info=False,
        directory='.'
):
    path = cuwb_data_path(
        filename_prefix,
        environment_name,
        start_time,
        end_time,
        environment_assignment_info,
        entity_assignment_info,
        directory
    )
    logger.info('Writing CUWB data to {}'.format(path))
    df.to_pickle(path)


def read_cuwb_data_pkl(
        filename_prefix,
        environment_name,
        start_time,
        end_time,
        environment_assignment_info=False,
        entity_assignment_info=False,
        directory='.'
):
    path = cuwb_data_path(
        filename_prefix,
        environment_name,
        start_time,
        end_time,
        environment_assignment_info,
        entity_assignment_info,
        directory
    )
    logger.info('Reading CUWB data from {}'.format(path))
    df = pd.read_pickle(path)
    return df


def cuwb_data_path(
        filename_prefix,
        environment_name,
        start_time,
        end_time,
        environment_assignment_info=False,
        entity_assignment_info=False,
        directory='.'
):
    start_time_string = "None"
    if start_time is not None:
        start_time_string = datetime_filename_format(start_time)
    end_time_string = "None"
    if end_time is not None:
        end_time_string = datetime_filename_format(end_time)
    filename = '-'.join([
        filename_prefix,
        environment_name,
        start_time_string,
        end_time_string
    ])
    if environment_assignment_info:
        filename = filename + '(env_assignments)'
    if entity_assignment_info:
        filename = filename + '(entity_assignments)'
    filename = filename + '.pkl'
    path = os.path.join(
        directory,
        filename
    )
    return path


def datetime_filename_format(timestamp):
    return timestamp.astimezone(
        datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')


def load_csv(
        path
):
    df = pd.read_csv(
        path
    )

    df.rename(columns={"start_time": "start_datetime", "end_time": "end_datetime"}, inplace=True)

    if 'start_datetime' in df.columns:
        df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    if 'end_datetime' in df.columns:
        df['end_datetime'] = pd.to_datetime(df['end_datetime'])

    if len(df['start_datetime']) == 0:
        return df

    # Recognize a supplied timezone but if missing assume UTC time was implied
    # TODO: handle per row timezone differences
    if df['start_datetime'][0].tzinfo is None or df['start_datetime'][0].tzinfo.utcoffset(
            df['start_datetime'][0]) is None:
        df['start_datetime'] = df['start_datetime'].dt.tz_localize("UTC")
        df['end_datetime'] = df['end_datetime'].dt.tz_localize("UTC")
    else:
        df['start_datetime'] = df['start_datetime'].dt.tz_convert('UTC')
        df['end_datetime'] = df['end_datetime'].dt.tz_convert('UTC')

    return df
