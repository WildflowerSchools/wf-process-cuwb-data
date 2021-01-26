import pandas as pd
import pytest

import json
import os


def cuwb_data_csv():
    """
    1 minute of CUWB data
    6 People
        3 w/o acceleration data
    5 Trays
    :return:
    """
    return os.path.dirname(os.path.realpath(__file__)) + '/fixtures/uwb.csv'


def cuwb_data_types_json():
    path = os.path.dirname(os.path.realpath(__file__)) + '/fixtures/uwb_data_types.json'
    with open(path) as data:
        return json.load(data)


@pytest.fixture(scope="module")
def cuwb_dataframe():
    df = pd.read_csv(cuwb_data_csv(), dtype=cuwb_data_types_json())
    df['timestamp'] = pd.to_datetime(df.timestamp)
    df.set_index('timestamp', inplace=True)
    return df
