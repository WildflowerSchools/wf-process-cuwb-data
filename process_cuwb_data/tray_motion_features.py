import pandas as pd
import numpy as np

from .tray_motion_filters import TrayMotionButterFiltFiltFilter, TrayMotionSavGolFilter
from .log import logger


class FeatureExtraction:
    def __init__(self, frequency="100ms", position_filter=TrayMotionButterFiltFiltFilter(),
                 velocity_filter=TrayMotionSavGolFilter()):
        self.frequency = frequency
        self.position_filter = position_filter
        self.velocity_filter = velocity_filter

    def extract_tray_motion_features_for_multiple_devices(self, df_position, df_acceleration):
        if ((len(df_position) == 0 or 'entity_type' not in df_position.columns) or
                (len(df_acceleration) == 0 or 'entity_type' not in df_acceleration.columns)):
            return None

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

        df['x_position_smoothed'] = self.position_filter.filter(
            series=df['x_meters'],
            btype=btype
        )
        df['y_position_smoothed'] = self.position_filter.filter(
            series=df['y_meters'],
            btype=btype
        )
        df['z_position_smoothed'] = self.position_filter.filter(
            series=df['z_meters'],
            btype=btype
        )
        # Old method of computing velocity, switched to savgol with deriv=1
        #df['x_velocity_smoothed']=df['x_position_smoothed'].diff().divide(df.index.to_series().diff().apply(lambda dt: dt.total_seconds()))
        #df['y_velocity_smoothed']=df['y_position_smoothed'].diff().divide(df.index.to_series().diff().apply(lambda dt: dt.total_seconds()))

        df['x_velocity_smoothed'] = self.velocity_filter.filter(
            df['x_position_smoothed'],
            deriv=1
        )
        df['y_velocity_smoothed'] = self.velocity_filter.filter(
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
