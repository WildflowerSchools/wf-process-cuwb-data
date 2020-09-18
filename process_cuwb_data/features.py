import pandas as pd
import numpy as np
import scipy.signal

from .log import logger


class ButterFiltFiltFilter:
    """
    Structure for storing Butter + FiltFilt signal filtering parameters

    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
    """

    def __init__(self, N=3, Wn=0.01, fs=10):
        """
        Parameters
        ----------
        N : int
            The order of the butter signal filter.
        Wn : array_like
            The critical frequency or frequencies for the butter signal filter.
        fs : float
            The sampling frequency for the butter signal filter.
        """
        self.N = N
        self.Wn = Wn
        self.fs = fs

    def filter(self, series, btype):
        butter_b, butter_a = scipy.signal.butter(N=self.N, Wn=self.Wn, fs=self.fs, btype=btype)
        series_filtered = scipy.signal.filtfilt(butter_b, butter_a, series)
        return series_filtered


class SavGolFilter:
    """
    Structure for storing SavGol signal filtering parameters

    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html
    """

    def __init__(self, window_length=31, polyorder=3, fs=10):
        """
        Parameters
        ----------
        window_length : int
            The length of the filter window (i.e., the number of coefficients)
        polyorder : int
            The order of the polynomial used to fit the samples
        fs : float
            The sampling frequency for the signal filter
        """
        self.window_length = window_length
        self.polyorder = polyorder
        self.delta = 1 / fs

    def filter(self, series, deriv):
        return scipy.signal.savgol_filter(
            series, deriv=deriv, window_length=self.window_length, polyorder=self.polyorder, delta=self.delta)


class PositionFilter:
    def __init__(self, butter_filt_filt=ButterFiltFiltFilter(), savgol=SavGolFilter()):
        if not isinstance(butter_filt_filt, ButterFiltFiltFilter):
            raise Exception("butter_filt_filt must be of type ButterFiltFiltFilter")

        if not isinstance(savgol, SavGolFilter):
            raise Exception("savgol must be of type SavGolFilter")

        self.butter_filt_filt = butter_filt_filt
        self.savgol = savgol


class FeatureExtraction:
    def __init__(self, frequency="100ms", position_filter=PositionFilter()):
        self.frequency = frequency
        self.position_filter = position_filter

    def extract_tray_motion_features_for_multiple_devices(self, df_position, df_acceleration):
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
            df_features = self.extract_tray_motion_features(
                df_position=df_position_reduced,
                df_acceleration=df_acceleration_reduced
            )
            df_features['device_id'] = device_id
            df_features = df_features.reindex(columns=[
                'device_id',
                'x_meters',
                'y_meters',
                'z_meters',
                'x_position_smoothed',
                'y_position_smoothed',
                'z_position_smoothed',
                'x_velocity_smoothed',
                'y_velocity_smoothed',
                'x_acceleration_normalized',
                'y_acceleration_normalized',
                'z_acceleration_normalized'
            ])
            df_dict[device_id] = df_features
        df_all = pd.concat(df_dict.values())
        return df_all

    def extract_tray_motion_features(self, df_position, df_acceleration):
        df_velocity_features = self.extract_velocity_features(
            df=df_position
        )
        df_acceleration_features = self.extract_acceleration_features(
            df=df_acceleration
        )
        df_features = df_velocity_features.join(
            df_acceleration_features, how='inner')
        df_features.dropna(inplace=True)
        return df_features

    def extract_velocity_features(self, df):
        df = df.copy()
        df = df.reindex(columns=[
            'x_meters',
            'y_meters',
            'z_meters'
        ])
        df = self.regularize_index(
            df
        )
        df = self.calculate_velocity_features(
            df=df
        )
        df = df.reindex(columns=[
            'x_meters',
            'y_meters',
            'z_meters',
            'x_position_smoothed',
            'y_position_smoothed',
            'z_position_smoothed',
            'x_velocity_smoothed',
            'y_velocity_smoothed'
        ])
        return df

    def extract_acceleration_features(self, df):
        df = df.copy()
        df = df.reindex(columns=[
            'x_gs',
            'y_gs',
            'z_gs'
        ])
        df = self.regularize_index(
            df
        )
        df = self.calculate_acceleration_features(
            df=df,
        )
        df = df.reindex(columns=[
            'x_acceleration_normalized',
            'y_acceleration_normalized',
            'z_acceleration_normalized',
        ])
        return df

    def regularize_index(self, df):
        df = df.copy()
        df = df.loc[~df.index.duplicated()].copy()
        start = df.index.min().floor(self.frequency)
        end = df.index.max().ceil(self.frequency)
        regular_index = pd.date_range(
            start=start,
            end=end,
            freq=self.frequency
        )
        df = df.reindex(df.index.union(regular_index))
        df = df.interpolate(method='time', limit_area='inside')
        df = df.reindex(regular_index).dropna()
        return df

    def calculate_velocity_features(self, df, inplace=False):
        btype = 'lowpass'
        if not inplace:
            df = df.copy()

        df['x_position_smoothed'] = self.position_filter.butter_filt_filt.filter(
            series=df['x_meters'],
            btype=btype
        )
        df['y_position_smoothed'] = self.position_filter.butter_filt_filt.filter(
            series=df['y_meters'],
            btype=btype
        )
        df['z_position_smoothed'] = self.position_filter.butter_filt_filt.filter(
            series=df['z_meters'],
            btype=btype
        )
        # Old method of computing velocity, switched to savgol with deriv=1
        #df['x_velocity_smoothed']=df['x_position_smoothed'].diff().divide(df.index.to_series().diff().apply(lambda dt: dt.total_seconds()))
        #df['y_velocity_smoothed']=df['y_position_smoothed'].diff().divide(df.index.to_series().diff().apply(lambda dt: dt.total_seconds()))

        df['x_velocity_smoothed'] = self.position_filter.savgol.filter(
            df['x_position_smoothed'],
            deriv=1
        )
        df['y_velocity_smoothed'] = self.position_filter.savgol.filter(
            df['y_position_smoothed'],
            deriv=1
        )

        if not inplace:
            return df

    def calculate_acceleration_features(self, df, inplace=False):
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


def combine_features_with_ground_truth_data(
    df_features,
    df_ground_truth,
    baseline_state='Not carried',
    inplace=False
):
    if not inplace:
        df_features = df_features.copy()
    df_features['ground_truth_state'] = baseline_state
    for index, row in df_ground_truth.iterrows():
        if row['ground_truth_state'] != baseline_state:
            df_features.loc[
                (
                    (df_features['device_id'] == row['device_id']) &
                    (df_features.index >= row['start_datetime']) &
                    (df_features.index <= row['end_datetime'])
                ),
                'ground_truth_state'
            ] = row['ground_truth_state']
    if not inplace:
        return df_features
