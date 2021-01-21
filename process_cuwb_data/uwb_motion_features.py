import pandas as pd
import numpy as np

from .uwb_motion_filters import TrayMotionButterFiltFiltFilter, TrayMotionSavGolFilter
from .log import logger


class FeatureExtraction:
    def __init__(self, frequency="100ms",
                 position_filter=TrayMotionButterFiltFiltFilter(useSosFiltFilt=True),
                 velocity_filter=TrayMotionSavGolFilter()):
        self.frequency = frequency
        self.position_filter = position_filter
        self.velocity_filter = velocity_filter

    def extract_motion_features_for_multiple_devices(self, df_position, df_acceleration, entity_type='all'):
        if ((len(df_position) == 0 or 'entity_type' not in df_position.columns) or
                (len(df_acceleration) == 0 or 'entity_type' not in df_acceleration.columns)):
            return None

        if entity_type is None:
            entity_type = 'all'

        pos_indx, acc_indx = slice(None), slice(None)
        if entity_type != 'all':
            pos_indx = df_position['entity_type'].str.lower() == entity_type.lower()
            acc_indx = df_acceleration['entity_type'].str.lower() == entity_type.lower()

        position_device_ids = df_position.loc[
            pos_indx,
            'device_id'
        ].unique().tolist()
        logger.info('Position data contains {} "{}" device IDs: {}'.format(
            len(position_device_ids),
            entity_type,
            position_device_ids,
        ))
        acceleration_device_ids = df_acceleration.loc[
            acc_indx,
            'device_id'
        ].unique().tolist()
        logger.info('Acceleration data contains {} "{}" device IDs: {}'.format(
            len(acceleration_device_ids),
            entity_type,
            acceleration_device_ids,
        ))
        device_ids = list(set(position_device_ids) & set(acceleration_device_ids))
        logger.info('Position and acceleration data contain {} common device IDs: {}'.format(
            len(device_ids),
            device_ids
        ))
        df_dict = dict()
        for device_id in device_ids:
            logger.info('Calculating motion features for device ID {}'.format(device_id))
            df_position_reduced = df_position.loc[
                df_position['device_id'] == device_id
            ].copy().sort_index()
            df_acceleration_reduced = df_acceleration.loc[
                df_acceleration['device_id'] == device_id
            ].copy().sort_index()
            df_features = self.extract_motion_features(
                df_position=df_position_reduced,
                df_acceleration=df_acceleration_reduced
            )
            df_features['device_id'] = device_id
            df_features = df_features.reindex(columns=[
                'device_id',
                'quality',
                'x_meters',
                'y_meters',
                'z_meters',
                'x_position_smoothed',
                'y_position_smoothed',
                'z_position_smoothed',
                'x_velocity_smoothed',
                'y_velocity_smoothed',
                'z_velocity_smoothed',
                # 'x_acceleration_dv_dt_smoothed',
                # 'y_acceleration_dv_dt_smoothed',
                # 'z_acceleration_dv_dt_smoothed',
                'x_acceleration_normalized',
                'y_acceleration_normalized',
                'z_acceleration_normalized',
                'x_acceleration_mean',
                'y_acceleration_mean',
                'z_acceleration_mean',
                'acceleration_average_mean',
                'x_acceleration_stddev',
                'y_acceleration_stddev',
                'z_acceleration_stddev',
                'acceleration_average_stddev',
                'x_acceleration_skew',
                'y_acceleration_skew',
                'z_acceleration_skew',
                'acceleration_average_skew',
                'x_acceleration_energy',
                'y_acceleration_energy',
                'z_acceleration_energy',
                'acceleration_average_energy',
                'x_y_acceleration_correlation',
                'x_z_acceleration_correlation',
                'y_z_acceleration_correlation',
                'x_acceleration_correlation_sum',
                'y_acceleration_correlation_sum',
                'z_acceleration_correlation_sum'
            ])
            df_dict[device_id] = df_features
        df_all = pd.concat(df_dict.values())
        return df_all

    def extract_tray_motion_features_for_multiple_devices(self, df_position, df_acceleration):
        return self.extract_motion_features_for_multiple_devices(df_position, df_acceleration, entity_type="tray")

    def extract_motion_features(self, df_position, df_acceleration):
        df_velocity_features = self.extract_velocity_features(
            df=df_position
        )
        df_acceleration_features = self.extract_acceleration_features(
            df=df_acceleration
        )
        df_features = df_velocity_features.join(
            df_acceleration_features, how='inner')

        df_quality = self.regularize_index(
            df_position[['quality']]
        )

        df_features = df_features.join(
            df_quality, how='inner')
        #df_features.dropna(inplace=True)
        df_features.fillna(df_features.mean(), inplace=True)
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
            'y_velocity_smoothed',
            'z_velocity_smoothed'
        ])
        return df

    def extract_acceleration_features(self, df):
        df = df.copy()
        df = df.reindex(columns=[
            # 'x_meters',
            # 'y_meters',
            # 'z_meters',
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
        df = df.drop(columns=['x_gs', 'y_gs', 'z_gs']).sort_index()
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
        if not inplace:
            df = df.copy()

        df['x_position_smoothed'] = self.position_filter.filter(
            series=df['x_meters']
        )
        df['y_position_smoothed'] = self.position_filter.filter(
            series=df['y_meters']
        )
        df['z_position_smoothed'] = self.position_filter.filter(
            series=df['z_meters']
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
        df['z_velocity_smoothed'] = self.velocity_filter.filter(
            df['z_position_smoothed'],
            deriv=1
        )

        if not inplace:
            return df

    def calculate_acceleration_features(self, df, inplace=False):
        if not inplace:
            df = df.copy()

        # x_smoothed = self.position_filter.filter(
        #     series=df['x_meters']
        # )
        # y_smoothed = self.position_filter.filter(
        #     series=df['y_meters']
        # )
        # z_smoothed = self.position_filter.filter(
        #     series=df['z_meters']
        # )
        #
        # df['x_acceleration_dv_dt_smoothed'] = self.velocity_filter.filter(
        #     x_smoothed,
        #     deriv=2
        # )
        # df['y_acceleration_dv_dt_smoothed'] = self.velocity_filter.filter(
        #     y_smoothed,
        #     deriv=2
        # )
        # df['z_acceleration_dv_dt_smoothed'] = self.velocity_filter.filter(
        #     z_smoothed,
        #     deriv=2
        # )

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

        window = int(1 / (pd.tseries.frequencies.to_offset(self.frequency).nanos / 1000000000))

        df['x_acceleration_mean'] = df['x_acceleration_normalized'].rolling(window=window, center=True).mean()
        df['y_acceleration_mean'] = df['y_acceleration_normalized'].rolling(window=window, center=True).mean()
        df['z_acceleration_mean'] = df['z_acceleration_normalized'].rolling(window=window, center=True).mean()
        df['acceleration_average_mean'] = df[['x_acceleration_mean', 'y_acceleration_mean', 'z_acceleration_mean']].mean(axis=1)
        df['x_acceleration_stddev'] = df['x_acceleration_normalized'].rolling(window=window, center=True).std()
        df['y_acceleration_stddev'] = df['y_acceleration_normalized'].rolling(window=window, center=True).std()
        df['z_acceleration_stddev'] = df['z_acceleration_normalized'].rolling(window=window, center=True).std()
        df['acceleration_average_stddev'] = df[['x_acceleration_stddev', 'y_acceleration_stddev', 'z_acceleration_stddev']].mean(axis=1)
        df['x_acceleration_skew'] = df['x_acceleration_normalized'].rolling(window=window, center=True).skew()
        df['y_acceleration_skew'] = df['y_acceleration_normalized'].rolling(window=window, center=True).skew()
        df['z_acceleration_skew'] = df['z_acceleration_normalized'].rolling(window=window, center=True).skew()
        df['acceleration_average_skew'] = df[['x_acceleration_skew', 'y_acceleration_skew', 'z_acceleration_skew']].mean(axis=1)
        df['x_acceleration_energy'] = df['x_acceleration_normalized'].pow(2).rolling(window=window, center=True).sum().div(window)
        df['y_acceleration_energy'] = df['y_acceleration_normalized'].pow(2).rolling(window=window, center=True).sum().div(window)
        df['z_acceleration_energy'] = df['z_acceleration_normalized'].pow(2).rolling(window=window, center=True).sum().div(window)
        df['acceleration_average_energy'] = df[['x_acceleration_energy', 'y_acceleration_energy', 'z_acceleration_energy']].mean(axis=1)
        df['x_y_acceleration_correlation'] = df['x_acceleration_normalized'].rolling(window=window).corr(df['y_acceleration_normalized'])
        df['x_z_acceleration_correlation'] = df['x_acceleration_normalized'].rolling(window=window).corr(df['z_acceleration_normalized'])
        df['y_z_acceleration_correlation'] = df['y_acceleration_normalized'].rolling(window=window).corr(df['z_acceleration_normalized'])
        df['x_acceleration_correlation_sum'] = df['x_y_acceleration_correlation'] + df['x_z_acceleration_correlation']
        df['y_acceleration_correlation_sum'] = df['x_y_acceleration_correlation'] + df['y_z_acceleration_correlation']
        df['z_acceleration_correlation_sum'] = df['x_z_acceleration_correlation'] + df['y_z_acceleration_correlation']

        if not inplace:
            return df
