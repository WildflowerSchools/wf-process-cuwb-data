import pandas as pd
import numpy as np
import scipy.signal
import logging

logger = logging.getLogger(__name__)

def extract_tray_motion_features_multiple_devices(
    df_position,
    df_acceleration,
    N,
    Wn,
    fs,
    freq='100ms'
):
    position_device_ids = df_position.loc[
        df_position['entity_type'] == 'Tray',
        'device_id'
    ].unique().tolist()
    logger.info('Position data contains {} tray device IDs: {}'.format(
        len(position_device_ids),
        position_device_ids,
    ))
    acceleration_device_ids = df_acceleration.loc[
        df_acceleration['entity_type'] == 'Tray',
        'device_id'
    ].unique().tolist()
    logger.info('Acceleration data contains {} tray device IDs: {}'.format(
        len(acceleration_device_ids),
        acceleration_device_ids,
    ))
    device_ids = list(set(position_device_ids) & set(acceleration_device_ids))
    logger.info('Position and acceleration data contain {} common tray device IDs: {}'.format(
        len(device_ids),
        device_ids
    ))
    df_dict = dict()
    for device_id in device_ids:
        logger.info('Calculating features for device ID {}'.format(device_id))
        df_position_reduced = df_position.loc[
            df_position['device_id'] == device_id
        ].copy().sort_index()
        df_acceleration_reduced = df_acceleration.loc[
            df_acceleration['device_id'] == device_id
        ].copy().sort_index()
        df_features = extract_tray_motion_features(
            df_position=df_position_reduced,
            df_acceleration=df_acceleration_reduced,
            N=N,
            Wn=Wn,
            fs=fs
        )
        df_features['device_id'] = device_id
        df_features = df_features.reindex(columns=[
            'device_id',
            'x_velocity_smoothed',
            'y_velocity_smoothed',
            'x_acceleration_normalized',
            'y_acceleration_normalized',
            'z_acceleration_normalized'
        ])
        df_dict[device_id] = df_features
    df_all = pd.concat(df_dict.values())
    return df_all

def extract_tray_motion_features(
    df_position,
    df_acceleration,
    N,
    Wn,
    fs,
    freq='100ms'
):
    df_velocity_features = extract_velocity_features(
        df=df_position,
        N=N,
        Wn=Wn,
        fs=fs,
        freq=freq
    )
    df_acceleration_features = extract_acceleration_features(
        df=df_acceleration,
        freq=freq
    )
    df_features = df_velocity_features.join(df_acceleration_features, how='inner')
    df_features.dropna(inplace=True)
    return df_features

def extract_velocity_features(
    df,
    N,
    Wn,
    fs,
    freq='100ms'
):
    df = df.copy()
    df = df.reindex(columns=[
        'x_meters',
        'y_meters',
        'z_meters'
    ])
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
    df = df.reindex(columns=[
        'x_gs',
        'y_gs',
        'z_gs'
    ])
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

def fetch_ground_truth_data(
    path,
    time_zone_name='US/Eastern',
    date_field_name = 'date',
    start_time_field_name='start_time',
    end_time_field_name='end_time'
):
    df = pd.read_csv(
        path,
        parse_dates={
            'start_datetime': [date_field_name, start_time_field_name],
            'end_datetime': [date_field_name, end_time_field_name]
        }
    )
    df['start_datetime'] = df['start_datetime'].dt.tz_localize(time_zone_name).dt.tz_convert('UTC')
    df['end_datetime'] = df['end_datetime'].dt.tz_localize(time_zone_name).dt.tz_convert('UTC')
    return df
