import pandas as pd
import numpy as np
import scipy.signal
import logging

logger = logging.getLogger(__name__)

def extract_velocity_features(
    df,
    N,
    Wn,
    fs,
    freq='100ms'
):
    df = df.copy()
    df = regularize_index(
        df,
        freq=freq
    )
    df = calculate_velocity_features(
        df=df,
        N=N,
        Wn=Wn,
        fs=fs
    )
    df = df.reindex(columns = [
        'x_velocity_smoothed',
        'y_velocity_smoothed'
    ])
    return df

def extract_acceleration_features(
    df,
    freq='100ms'
):
    df = df.copy()
    df = regularize_index(
        df,
        freq=freq
    )
    df = calculate_acceleration_features(
        df=df,
    )
    df = df.reindex(columns = [
        'x_acceleration_normalized',
        'y_acceleration_normalized',
        'z_acceleration_normalized',
    ])
    return df

def regularize_index(
    df,
    freq='100ms'
):
    df = df.copy()
    df = df.loc[~df.index.duplicated()].copy()
    start = df.index.min().floor(freq)
    end = df.index.max().ceil(freq)
    regular_index = pd.date_range(
        start=start,
        end=end,
        freq=freq
    )
    df = df.reindex(df.index.union(regular_index))
    df = df.interpolate(method='time', limit_area='inside')
    df = df.reindex(regular_index).dropna()
    return df

def calculate_velocity_features(
    df,
    N,
    Wn,
    fs,
    inplace=False
):
    btype='lowpass'
    if not inplace:
        df = df.copy()
    df['x_position_smoothed'] = filter_butter_filtfilt(
        series=df['x_meters'],
        N=N,
        Wn=Wn,
        fs=fs,
        btype=btype
    )
    df['y_position_smoothed'] = filter_butter_filtfilt(
        series=df['y_meters'],
        N=N,
        Wn=Wn,
        fs=fs,
        btype=btype
    )
    df['x_velocity_smoothed']=df['x_position_smoothed'].diff().divide(df.index.to_series().diff().apply(lambda dt: dt.total_seconds()))
    df['y_velocity_smoothed']=df['y_position_smoothed'].diff().divide(df.index.to_series().diff().apply(lambda dt: dt.total_seconds()))
    if not inplace:
        return df

def filter_butter_filtfilt(
    series,
    N,
    Wn,
    fs,
    btype='lowpass'
):
    butter_b, butter_a = scipy.signal.butter(N=N, Wn=Wn, btype=btype, fs=fs)
    series_filtered = scipy.signal.filtfilt(butter_b, butter_a, series)
    return series_filtered

def calculate_acceleration_features(
    df,
    inplace=False
):
    if not inplace:
        df = df.copy()
    df['x_acceleration_normalized'] = np.subtract(
        df['x_gs'],
        df['x_gs'].mean()
    )
    df['y_acceleration_normalized'] = np.subtract(
        df['y_gs'],
        df['y_gs'].mean()
    )
    df['z_acceleration_normalized'] = np.subtract(
        df['z_gs'],
        df['z_gs'].mean()
    )
    if not inplace:
        return df
