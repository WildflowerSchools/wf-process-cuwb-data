import multiprocessing
import sys
import time
from functools import partial

import pandas as pd
import torch

from .utils.log import logger
from .utils.util import map_column_name_to_dimension_space

DIMENSIONS_WHEN_COMPUTING_DEVICE_TO_DEVICE_DISTANCES = 2


def device_to_device_cdist_iterable(idx, df_devices_a, df_devices_b, lsuffix, rsuffix, v_count, v_start, lock, size):
    """
    Background runnable function to compute distances between devices.

    :param idx: Dataframe lookup index (time based)
    :param df_devices_a: Devices dataframe A
    :param df_devices_b: Devices dataframe B
    :param v_count: Iteration count
    :param v_start: Timer for logging progress
    :param lock: Multiprocessing lock for variable manipulation
    :param size: Total number of records for processing
    :return: Dataframe of joined devices with computed distance
    """
    if lsuffix is None:
        lsuffix = "_a"

    if rsuffix is None:
        rsuffix = "_b"

    with lock:
        if v_count.value % 1000 == 0:
            logger.info(
                "Computing device <-> device distances: Processed {}/{} - Time {}s".format(
                    v_count.value, size, time.time() - v_start.value
                )
            )
            sys.stdout.flush()
            v_start.value = time.time()

        v_count.value += 1

    if idx not in df_devices_a.index:
        return None
    df_device_a_by_idx = df_devices_a.loc[[idx]]

    if idx not in df_devices_b.index:
        return None
    df_device_b_by_idx = df_devices_b.loc[[idx]]

    position_cols = map_column_name_to_dimension_space("position", DIMENSIONS_WHEN_COMPUTING_DEVICE_TO_DEVICE_DISTANCES)

    df_device_to_device_join = df_device_a_by_idx.join(
        df_device_b_by_idx, how="inner", lsuffix=lsuffix, rsuffix=rsuffix
    )
    distances = torch.cdist(
        torch.tensor(df_device_a_by_idx[position_cols].to_numpy()),
        torch.tensor(df_device_b_by_idx[position_cols].to_numpy()),
    )

    return df_device_to_device_join.assign(device_to_device_distance=distances.flatten())


def generate_device_to_device_distances(df_device_features_a, df_device_features_b, lsuffix="_a", rsuffix="_b"):
    """
    Use multi-processing to generate distances between candidate devices
    across all recorded features (computationally heavy because distance is generated
    for every timestamp between every pair)

    :param df_device_features_a:
    :param df_device_features_b:
    :return:
    """
    p = multiprocessing.Pool()
    m = multiprocessing.Manager()

    lock = m.Lock()
    start = time.time()
    v_count = m.Value("i", 0)  # Keep track of iterations
    v_start = m.Value("f", start)  # Share timer object
    time_indexes = df_device_features_a.index.unique(level=0)

    # This computes distances between all devices
    df_device_to_device_distances = pd.concat(
        p.map(
            partial(
                device_to_device_cdist_iterable,
                df_devices_a=df_device_features_a,
                df_devices_b=df_device_features_b,
                lsuffix=lsuffix,
                rsuffix=rsuffix,
                v_count=v_count,
                v_start=v_start,
                lock=lock,
                size=len(time_indexes),
            ),
            time_indexes,
        )
    )
    logger.info(
        "Finished computing device <-> device distances: {}/{} - Total time: {}s".format(
            len(time_indexes), len(time_indexes), time.time() - start
        )
    )

    p.close()
    p.join()

    return df_device_to_device_distances
