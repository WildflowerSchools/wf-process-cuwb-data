import pandas as pd
import numpy as np

from .utils.util import filter_entity_type
from .uwb_motion_filters import TrayMotionButterFiltFiltFilter, TrayMotionSavGolFilter
from process_cuwb_data.utils.log import logger


class FeatureExtraction:
    def __init__(self, frequency="100ms",
                 position_filter=TrayMotionButterFiltFiltFilter(useSosFiltFilt=True),
                 velocity_filter=TrayMotionSavGolFilter()):
        self.frequency = frequency
        self.position_filter = position_filter
        self.velocity_filter = velocity_filter

    VELOCITY_COLUMNS = [
        'quality',
        'x_mps',
        'y_mps',
        'z_mps',
        'x_position_smoothed',
        'y_position_smoothed',
        'z_position_smoothed',
        'x_velocity_smoothed',
        'y_velocity_smoothed',
        'z_velocity_smoothed',
        'x_velocity_smoothed_magnitude',
        'y_velocity_smoothed_magnitude',
        'z_velocity_smoothed_magnitude',
        'velocity_vector_magnitude',
        'velocity_vector_magnitude_xy',
        'x_velocity_mean',
        'y_velocity_mean',
        'z_velocity_mean',
        'velocity_average_mean',
        'velocity_vector_magnitude_mean',
        'velocity_vector_magnitude_mean_xy',
        'x_velocity_stddev',
        'y_velocity_stddev',
        'z_velocity_stddev',
        'velocity_average_stddev',
        'velocity_vector_magnitude_stddev',
        'velocity_vector_magnitude_stddev_xy',
        'x_velocity_skew',
        'y_velocity_skew',
        'z_velocity_skew',
        'velocity_average_skew',
        'velocity_vector_magnitude_skew',
        'velocity_vector_magnitude_skew_xy',
        'x_velocity_variance',
        'y_velocity_variance',
        'z_velocity_variance',
        'velocity_average_variance',
        'velocity_vector_magnitude_variance',
        'velocity_vector_magnitude_variance_xy',
        'x_velocity_kurtosis',
        'y_velocity_kurtosis',
        'z_velocity_kurtosis',
        'velocity_average_kurtosis',
        'velocity_vector_magnitude_kurtosis',
        'velocity_vector_magnitude_kurtosis_xy',
        'x_y_velocity_correlation',
        'x_z_velocity_correlation',
        'y_z_velocity_correlation',
        'x_velocity_correlation_sum',
        'y_velocity_correlation_sum',
        'z_velocity_correlation_sum'
    ]

    ACCELERATION_COLUMNS = [
        'x_acceleration_normalized',
        'y_acceleration_normalized',
        'z_acceleration_normalized',
        'acceleration_vector_magnitude',
        'x_acceleration_mean',
        'y_acceleration_mean',
        'z_acceleration_mean',
        'acceleration_average_mean',
        'acceleration_vector_magnitude_mean',
        'x_acceleration_sum',
        'y_acceleration_sum',
        'z_acceleration_sum',
        'acceleration_average_sum',
        'acceleration_vector_magnitude_sum',
        'x_acceleration_min',
        'y_acceleration_min',
        'z_acceleration_min',
        'acceleration_average_min',
        'acceleration_vector_magnitude_min',
        'x_acceleration_max',
        'y_acceleration_max',
        'z_acceleration_max',
        'acceleration_average_max',
        'acceleration_vector_magnitude_max',
        'x_acceleration_stddev',
        'y_acceleration_stddev',
        'z_acceleration_stddev',
        'acceleration_average_stddev',
        'acceleration_vector_magnitude_stddev',
        'x_acceleration_skew',
        'y_acceleration_skew',
        'z_acceleration_skew',
        'acceleration_average_skew',
        'acceleration_vector_magnitude_skew',
        'x_acceleration_variance',
        'y_acceleration_variance',
        'z_acceleration_variance',
        'acceleration_average_variance',
        'acceleration_vector_magnitude_variance',
        'x_acceleration_kurtosis',
        'y_acceleration_kurtosis',
        'z_acceleration_kurtosis',
        'acceleration_average_kurtosis',
        'acceleration_vector_magnitude_kurtosis',
        'x_acceleration_energy',
        'y_acceleration_energy',
        'z_acceleration_energy',
        'acceleration_average_energy',
        'acceleration_vector_magnitude_energy',
        'x_y_acceleration_correlation',
        'x_z_acceleration_correlation',
        'y_z_acceleration_correlation',
        'x_acceleration_correlation_sum',
        'y_acceleration_correlation_sum',
        'z_acceleration_correlation_sum'
    ]

    GYROSCOPE_COLUMNS = [
        'x_dps',
        'y_dps',
        'z_dps'
    ]

    MAGNETOMETER_COLUMNS = [
        'x_μT',
        'y_μT',
        'z_μT'
    ]

    ALL_FEATURE_COLUMNS = [
        *VELOCITY_COLUMNS,
        *ACCELERATION_COLUMNS,
        *GYROSCOPE_COLUMNS,
        *MAGNETOMETER_COLUMNS
    ]

    def extract_motion_features_for_multiple_devices(
            self,
            df_position=None,
            df_acceleration=None,
            df_gyroscope=None,
            df_magnetometer=None,
            entity_type='all',
            fillna=None,
            join='outer'):
        if ((df_position is None and df_acceleration is None) or
                (
            (df_position is not None and len(df_position) == 0) and
            (df_acceleration is not None and len(df_acceleration) == 0) and
            (df_gyroscope is not None and len(df_gyroscope) == 0) and
            (df_magnetometer is not None and len(df_magnetometer) == 0)
        ) or
                (
            (df_position is not None and 'entity_type' not in df_position.columns) or
            (df_acceleration is not None and 'entity_type' not in df_acceleration.columns) or
            (df_gyroscope is not None and 'entity_type' not in df_gyroscope.columns) or
            (df_magnetometer is not None and 'entity_type' not in df_magnetometer.columns)
        )):
            return None

        if entity_type is None:
            entity_type = 'all'

        pos_indx, acc_indx, gyr_indx, mag_indx = slice(None), slice(None), slice(None), slice(None)
        if entity_type != 'all':
            if df_position is not None:
                pos_indx = df_position['entity_type'].str.lower() == entity_type.lower()
            if df_acceleration is not None:
                acc_indx = df_acceleration['entity_type'].str.lower() == entity_type.lower()
            if df_gyroscope is not None:
                gyr_indx = df_gyroscope['entity_type'].str.lower() == entity_type.lower()
            if df_magnetometer is not None:
                mag_indx = df_magnetometer['entity_type'].str.lower() == entity_type.lower()

        position_device_ids = []
        if df_position is not None:
            position_device_ids = df_position.loc[
                pos_indx,
                'device_id'
            ].unique().tolist()
            logger.info('Position data contains {} "{}" device IDs: {}'.format(
                len(position_device_ids),
                entity_type,
                position_device_ids,
            ))

        acceleration_device_ids = []
        if df_acceleration is not None:
            acceleration_device_ids = df_acceleration.loc[
                acc_indx,
                'device_id'
            ].unique().tolist()
            logger.info('Acceleration data contains {} "{}" device IDs: {}'.format(
                len(acceleration_device_ids),
                entity_type,
                acceleration_device_ids,
            ))

        gyroscope_device_ids = []
        if df_gyroscope is not None:
            gyroscope_device_ids = df_gyroscope.loc[
                gyr_indx,
                'device_id'
            ].unique().tolist()
            logger.info('Gyroscope data contains {} "{}" device IDs: {}'.format(
                len(gyroscope_device_ids),
                entity_type,
                gyroscope_device_ids,
            ))

        magnetometer_device_ids = []
        if df_magnetometer is not None:
            magnetometer_device_ids = df_magnetometer.loc[
                mag_indx,
                'device_id'
            ].unique().tolist()
            logger.info('Magnetometer data contains {} "{}" device IDs: {}'.format(
                len(magnetometer_device_ids),
                entity_type,
                magnetometer_device_ids,
            ))

        all_device_ids = list(set(position_device_ids) |
                              set(acceleration_device_ids) |
                              set(gyroscope_device_ids) |
                              set(magnetometer_device_ids))

        df_dict = dict()
        for device_id in all_device_ids:
            logger.info('Calculating motion features for device ID {}'.format(device_id))

            df_position_for_device = None
            if df_position is not None:
                df_position_for_device = df_position.loc[
                    df_position['device_id'] == device_id
                ].copy().sort_index()

            df_acceleration_for_device = None
            if df_acceleration is not None:
                df_acceleration_for_device = df_acceleration.loc[
                    df_acceleration['device_id'] == device_id
                ].copy().sort_index()

            df_gyroscope_for_device = None
            if df_gyroscope is not None:
                df_gyroscope_for_device = df_gyroscope.loc[
                    df_gyroscope['device_id'] == device_id
                ].copy().sort_index()

            df_magnetomter_for_device = None
            if df_magnetometer is not None:
                df_magnetomter_for_device = df_magnetometer.loc[
                    df_magnetometer['device_id'] == device_id
                ].copy().sort_index()

            df_features = self.extract_motion_features(
                df_position=df_position_for_device,
                df_acceleration=df_acceleration_for_device,
                df_gyroscope=df_gyroscope_for_device,
                df_magnetomter=df_magnetomter_for_device,
                fillna=fillna,
                join=join
            )
            df_features['device_id'] = device_id

            df_dict[device_id] = df_features

        df_all = pd.concat(df_dict.values())
        return df_all

    def extract_tray_motion_features_for_multiple_devices(
            self, df_position, df_acceleration, df_gyroscope, df_magnetometer):
        return self.extract_motion_features_for_multiple_devices(df_position=df_position,
                                                                 df_acceleration=df_acceleration,
                                                                 df_gyroscope=df_gyroscope,
                                                                 df_magnetometer=df_magnetometer,
                                                                 entity_type="tray")

    def extract_motion_features(self,
                                df_position,
                                df_acceleration,
                                df_gyroscope=None,
                                df_magnetomter=None,
                                fillna=None,
                                join='outer'):
        df_velocity_features = pd.DataFrame(columns=FeatureExtraction.VELOCITY_COLUMNS)
        if df_position is not None:
            df_velocity_features = self.extract_velocity_features(
                df=df_position
            )

            df_quality = self.regularize_index(
                df_position[['quality']]
            )

            df_velocity_features = df_velocity_features.join(
                df_quality, how='inner')

        df_acceleration_features = pd.DataFrame(columns=FeatureExtraction.ACCELERATION_COLUMNS)
        if df_acceleration is not None:
            df_acceleration_features = self.extract_acceleration_features(
                df=df_acceleration
            )

        df_gyroscope_features = pd.DataFrame(columns=FeatureExtraction.GYROSCOPE_COLUMNS)
        if df_gyroscope is not None:
            df_gyroscope_features = self.extract_gyroscope_features(
                df=df_gyroscope
            )

        df_magnetometer_features = pd.DataFrame(columns=FeatureExtraction.MAGNETOMETER_COLUMNS)
        if df_magnetomter is not None:
            df_magnetometer_features = self.extract_magnetometer_features(
                df=df_magnetomter
            )

        df_features = df_velocity_features\
            .join(
                df_acceleration_features,
                how=join)\
            .join(
                df_gyroscope_features,
                how=join)\
            .join(
                df_magnetometer_features,
                how=join)

        df_features = df_features.reindex(columns=[
            'device_id',
            *FeatureExtraction.VELOCITY_COLUMNS,
            *FeatureExtraction.ACCELERATION_COLUMNS,
            *FeatureExtraction.GYROSCOPE_COLUMNS,
            *FeatureExtraction.MAGNETOMETER_COLUMNS
        ])

        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)

        if fillna == 'average':
            df_features.fillna(df_features.mean(), inplace=True)
        elif fillna == 'drop':
            df_features.dropna(inplace=True)
        elif fillna == 'pad':
            df_features.fillna(method="pad", inplace=True)
        elif fillna == 'forward_backward':
            df_features = df_features.fillna(method='ffill').fillna(method='bfill')
        elif fillna == 'interpolate':  # linear interpolation, use bfill to fill in nan's at front of dataframe
            df_features = df_features.interpolate().fillna(method='bfill')

        return df_features

    def extract_velocity_features(self, df):
        df = df.copy()

        if 'x' in df.columns:
            df.rename(
                columns={
                    'x': 'x_mps',
                    'y': 'y_mps',
                    'z': 'z_mps'
                },
                inplace=True
            )

        if 'x_meters' in df.columns:
            df.rename(
                columns={
                    'x_meters': 'x_mps',
                    'y_meters': 'y_mps',
                    'z_meters': 'z_mps'
                },
                inplace=True
            )

        df = df.reindex(columns=['x_mps', 'y_mps', 'z_mps'])
        df = self.regularize_index(
            df
        )
        df = self.calculate_velocity_features(
            df=df
        )
        df = df.sort_index()
        return df

    def extract_acceleration_features(self, df):
        df = df.copy()
        if 'x' in df.columns:
            df.rename(
                columns={
                    'x': 'x_gs',
                    'y': 'y_gs',
                    'z': 'z_gs'
                },
                inplace=True
            )

        df = df.reindex(columns=['x_gs', 'y_gs', 'z_gs'])
        df = self.regularize_index(
            df
        )
        df = self.calculate_acceleration_features(
            df=df,
        )
        df = df.drop(columns=['x_gs', 'y_gs', 'z_gs']).sort_index()
        return df

    def extract_gyroscope_features(self, df):
        df = df.copy()

        if 'x' in df.columns:
            df.rename(
                columns={
                    'x': 'x_dps',
                    'y': 'y_dps',
                    'z': 'z_dps'
                },
                inplace=True
            )

        df = df.reindex(columns=['x_dps', 'y_dps', 'z_dps'])
        df = self.regularize_index(
            df
        )
        return df

    def extract_magnetometer_features(self, df):
        df = df.copy()

        if 'x' in df.columns:
            df.rename(
                columns={
                    'x': 'x_μT',
                    'y': 'y_μT',
                    'z': 'z_μT'
                },
                inplace=True
            )

        df = df.reindex(columns=['x_μT', 'y_μT', 'z_μT'])
        df = self.regularize_index(
            df
        )
        return df

    def regularize_index(self, df):
        df = df.copy()

        if len(df) == 0:
            return df

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

        if 'x' in df.columns:
            df.rename(
                columns={
                    'x': 'x_mps',
                    'y': 'y_mps',
                    'z': 'z_mps'
                },
                inplace=True
            )

        df['x_position_smoothed'] = self.position_filter.filter(
            series=df['x_mps']
        )
        df['y_position_smoothed'] = self.position_filter.filter(
            series=df['y_mps']
        )
        df['z_position_smoothed'] = self.position_filter.filter(
            series=df['z_mps']
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

        df['x_velocity_smoothed_magnitude'] = df['x_velocity_smoothed'].abs()
        df['y_velocity_smoothed_magnitude'] = df['y_velocity_smoothed'].abs()
        df['z_velocity_smoothed_magnitude'] = df['z_velocity_smoothed'].abs()

        df['velocity_vector_magnitude'] = df[['x_velocity_smoothed_magnitude',
                                              'y_velocity_smoothed_magnitude', 'z_velocity_smoothed_magnitude']].pow(2).sum(axis=1).pow(.5)

        df['velocity_vector_magnitude_xy'] = df[['x_velocity_smoothed_magnitude',
                                                 'y_velocity_smoothed_magnitude']].pow(2).sum(axis=1).pow(.5)

        window = int(1 / (pd.tseries.frequencies.to_offset(self.frequency).nanos / 1000000000))

        df['x_velocity_mean'] = df['x_velocity_smoothed_magnitude'].rolling(window=window, center=True).mean()
        df['y_velocity_mean'] = df['y_velocity_smoothed_magnitude'].rolling(window=window, center=True).mean()
        df['z_velocity_mean'] = df['z_velocity_smoothed_magnitude'].rolling(window=window, center=True).mean()
        df['velocity_average_mean'] = df[['x_velocity_mean', 'y_velocity_mean', 'z_velocity_mean']].mean(axis=1)

        df['velocity_vector_magnitude_mean'] = df['velocity_vector_magnitude'].rolling(
            window=window, center=True).mean()
        df['velocity_vector_magnitude_mean_xy'] = df['velocity_vector_magnitude_xy'].rolling(
            window=window, center=True).mean()

        df['x_velocity_stddev'] = df['x_velocity_smoothed_magnitude'].rolling(window=window, center=True).std()
        df['y_velocity_stddev'] = df['y_velocity_smoothed_magnitude'].rolling(window=window, center=True).std()
        df['z_velocity_stddev'] = df['z_velocity_smoothed_magnitude'].rolling(window=window, center=True).std()
        df['velocity_average_stddev'] = df[['x_velocity_stddev', 'y_velocity_stddev', 'z_velocity_stddev']].mean(axis=1)

        df['velocity_vector_magnitude_stddev'] = df['velocity_vector_magnitude'].rolling(
            window=window, center=True).std()
        df['velocity_vector_magnitude_stddev_xy'] = df['velocity_vector_magnitude_xy'].rolling(
            window=window, center=True).std()

        df['x_velocity_skew'] = df['x_velocity_smoothed_magnitude'].rolling(window=window, center=True).skew()
        df['y_velocity_skew'] = df['y_velocity_smoothed_magnitude'].rolling(window=window, center=True).skew()
        df['z_velocity_skew'] = df['z_velocity_smoothed_magnitude'].rolling(window=window, center=True).skew()
        df['velocity_average_skew'] = df[['x_velocity_skew', 'y_velocity_skew', 'z_velocity_skew']].mean(axis=1)

        df['velocity_vector_magnitude_skew'] = df['velocity_vector_magnitude'].rolling(
            window=window, center=True).skew()
        df['velocity_vector_magnitude_skew_xy'] = df['velocity_vector_magnitude_xy'].rolling(
            window=window, center=True).skew()

        df['x_velocity_variance'] = df['x_velocity_smoothed_magnitude'].rolling(window=window, center=True).var()
        df['y_velocity_variance'] = df['y_velocity_smoothed_magnitude'].rolling(window=window, center=True).var()
        df['z_velocity_variance'] = df['z_velocity_smoothed_magnitude'].rolling(window=window, center=True).var()
        df['velocity_average_variance'] = df[['x_velocity_variance',
                                              'y_velocity_variance', 'z_velocity_variance']].mean(axis=1)

        df['velocity_vector_magnitude_variance'] = df['velocity_vector_magnitude'].rolling(
            window=window, center=True).var()
        df['velocity_vector_magnitude_variance_xy'] = df['velocity_vector_magnitude_xy'].rolling(
            window=window, center=True).var()

        df['x_velocity_kurtosis'] = df['x_velocity_smoothed_magnitude'].rolling(window=window, center=True).kurt()
        df['y_velocity_kurtosis'] = df['y_velocity_smoothed_magnitude'].rolling(window=window, center=True).kurt()
        df['z_velocity_kurtosis'] = df['z_velocity_smoothed_magnitude'].rolling(window=window, center=True).kurt()
        df['velocity_average_kurtosis'] = df[['x_velocity_kurtosis',
                                              'y_velocity_kurtosis', 'z_velocity_kurtosis']].mean(axis=1)

        df['velocity_vector_magnitude_kurtosis'] = df['velocity_vector_magnitude'].rolling(
            window=window, center=True).kurt()
        df['velocity_vector_magnitude_kurtosis_xy'] = df['velocity_vector_magnitude_xy'].rolling(
            window=window, center=True).kurt()

        df['x_y_velocity_correlation'] = df['x_velocity_smoothed_magnitude'].rolling(
            window=window).corr(df['y_velocity_smoothed_magnitude'])
        df['x_z_velocity_correlation'] = df['x_velocity_smoothed_magnitude'].rolling(
            window=window).corr(df['z_velocity_smoothed_magnitude'])
        df['y_z_velocity_correlation'] = df['y_velocity_smoothed_magnitude'].rolling(
            window=window).corr(df['z_velocity_smoothed_magnitude'])
        df['x_velocity_correlation_sum'] = df['x_y_velocity_correlation'] + df['x_z_velocity_correlation']
        df['y_velocity_correlation_sum'] = df['x_y_velocity_correlation'] + df['y_z_velocity_correlation']
        df['z_velocity_correlation_sum'] = df['x_z_velocity_correlation'] + df['y_z_velocity_correlation']

        if not inplace:
            return df

    def calculate_acceleration_features(self, df, inplace=False):
        if not inplace:
            df = df.copy()

        if 'x' in df.columns:
            df.rename(
                columns={
                    'x': 'x_gs',
                    'y': 'y_gs',
                    'z': 'z_gs'
                },
                inplace=True
            )

        df['x_acceleration_normalized'] = np.absolute(np.subtract(
            df['x_gs'],
            df['x_gs'].mean()
        ))
        df['y_acceleration_normalized'] = np.absolute(np.subtract(
            df['y_gs'],
            df['y_gs'].mean()
        ))
        df['z_acceleration_normalized'] = np.absolute(np.subtract(
            df['z_gs'],
            df['z_gs'].mean()
        ))

        df['acceleration_vector_magnitude'] = df[['x_acceleration_normalized',
                                                  'y_acceleration_normalized', 'z_acceleration_normalized']].pow(2).sum(axis=1).pow(.5)

        window = int(1 / (pd.tseries.frequencies.to_offset(self.frequency).nanos / 1000000000))

        df['x_acceleration_mean'] = df['x_acceleration_normalized'].rolling(window=window, center=True).mean()
        df['y_acceleration_mean'] = df['y_acceleration_normalized'].rolling(window=window, center=True).mean()
        df['z_acceleration_mean'] = df['z_acceleration_normalized'].rolling(window=window, center=True).mean()
        df['acceleration_average_mean'] = df[['x_acceleration_mean',
                                              'y_acceleration_mean', 'z_acceleration_mean']].mean(axis=1)

        df['acceleration_vector_magnitude_mean'] = df['acceleration_vector_magnitude'].rolling(
            window=window, center=True).mean()

        df['x_acceleration_sum'] = df['x_acceleration_normalized'].rolling(window=window, center=True).sum()
        df['y_acceleration_sum'] = df['y_acceleration_normalized'].rolling(window=window, center=True).sum()
        df['z_acceleration_sum'] = df['z_acceleration_normalized'].rolling(window=window, center=True).sum()
        df['acceleration_average_sum'] = df[['x_acceleration_sum',
                                             'y_acceleration_sum', 'z_acceleration_sum']].mean(axis=1)

        df['acceleration_vector_magnitude_sum'] = df['acceleration_vector_magnitude'].rolling(
            window=window, center=True).sum()

        df['x_acceleration_min'] = df['x_acceleration_normalized'].rolling(window=window, center=True).min()
        df['y_acceleration_min'] = df['y_acceleration_normalized'].rolling(window=window, center=True).min()
        df['z_acceleration_min'] = df['z_acceleration_normalized'].rolling(window=window, center=True).min()
        df['acceleration_average_min'] = df[['x_acceleration_min',
                                             'y_acceleration_min', 'z_acceleration_min']].mean(axis=1)

        df['acceleration_vector_magnitude_min'] = df['acceleration_vector_magnitude'].rolling(
            window=window, center=True).min()

        df['x_acceleration_max'] = df['x_acceleration_normalized'].rolling(window=window, center=True).max()
        df['y_acceleration_max'] = df['y_acceleration_normalized'].rolling(window=window, center=True).max()
        df['z_acceleration_max'] = df['z_acceleration_normalized'].rolling(window=window, center=True).max()
        df['acceleration_average_max'] = df[['x_acceleration_max',
                                             'y_acceleration_max', 'z_acceleration_max']].mean(axis=1)

        df['acceleration_vector_magnitude_max'] = df['acceleration_vector_magnitude'].rolling(
            window=window, center=True).max()

        df['x_acceleration_stddev'] = df['x_acceleration_normalized'].rolling(window=window, center=True).std()
        df['y_acceleration_stddev'] = df['y_acceleration_normalized'].rolling(window=window, center=True).std()
        df['z_acceleration_stddev'] = df['z_acceleration_normalized'].rolling(window=window, center=True).std()
        df['acceleration_average_stddev'] = df[['x_acceleration_stddev',
                                                'y_acceleration_stddev', 'z_acceleration_stddev']].mean(axis=1)

        df['acceleration_vector_magnitude_stddev'] = df['acceleration_vector_magnitude'].rolling(
            window=window, center=True).std()

        df['x_acceleration_skew'] = df['x_acceleration_normalized'].rolling(window=window, center=True).skew()
        df['y_acceleration_skew'] = df['y_acceleration_normalized'].rolling(window=window, center=True).skew()
        df['z_acceleration_skew'] = df['z_acceleration_normalized'].rolling(window=window, center=True).skew()
        df['acceleration_average_skew'] = df[['x_acceleration_skew',
                                              'y_acceleration_skew', 'z_acceleration_skew']].mean(axis=1)

        df['acceleration_vector_magnitude_skew'] = df['acceleration_vector_magnitude'].rolling(
            window=window, center=True).skew()

        df['x_acceleration_variance'] = df['x_acceleration_normalized'].rolling(window=window, center=True).var()
        df['y_acceleration_variance'] = df['y_acceleration_normalized'].rolling(window=window, center=True).var()
        df['z_acceleration_variance'] = df['z_acceleration_normalized'].rolling(window=window, center=True).var()
        df['acceleration_average_variance'] = df[['x_acceleration_variance',
                                                  'y_acceleration_variance', 'z_acceleration_variance']].mean(axis=1)

        df['acceleration_vector_magnitude_variance'] = df['acceleration_vector_magnitude'].rolling(
            window=window, center=True).var()

        df['x_acceleration_kurtosis'] = df['x_acceleration_normalized'].rolling(window=window, center=True).kurt()
        df['y_acceleration_kurtosis'] = df['y_acceleration_normalized'].rolling(window=window, center=True).kurt()
        df['z_acceleration_kurtosis'] = df['z_acceleration_normalized'].rolling(window=window, center=True).kurt()
        df['acceleration_average_kurtosis'] = df[['x_acceleration_kurtosis',
                                                  'y_acceleration_kurtosis', 'z_acceleration_kurtosis']].mean(axis=1)

        df['acceleration_vector_magnitude_kurtosis'] = df['acceleration_vector_magnitude'].rolling(
            window=window, center=True).kurt()

        df['x_acceleration_energy'] = df['x_acceleration_normalized'].pow(
            2).rolling(window=window, center=True).sum().div(window)
        df['y_acceleration_energy'] = df['y_acceleration_normalized'].pow(
            2).rolling(window=window, center=True).sum().div(window)
        df['z_acceleration_energy'] = df['z_acceleration_normalized'].pow(
            2).rolling(window=window, center=True).sum().div(window)
        df['acceleration_average_energy'] = df[['x_acceleration_energy',
                                                'y_acceleration_energy', 'z_acceleration_energy']].mean(axis=1)

        df['acceleration_vector_magnitude_energy'] = df['acceleration_vector_magnitude'].pow(
            2).rolling(window=window, center=True).sum().div(window)

        df['x_y_acceleration_correlation'] = df['x_acceleration_normalized'].rolling(
            window=window).corr(df['y_acceleration_normalized'])
        df['x_z_acceleration_correlation'] = df['x_acceleration_normalized'].rolling(
            window=window).corr(df['z_acceleration_normalized'])
        df['y_z_acceleration_correlation'] = df['y_acceleration_normalized'].rolling(
            window=window).corr(df['z_acceleration_normalized'])
        df['x_acceleration_correlation_sum'] = df['x_y_acceleration_correlation'] + df['x_z_acceleration_correlation']
        df['y_acceleration_correlation_sum'] = df['x_y_acceleration_correlation'] + df['y_z_acceleration_correlation']
        df['z_acceleration_correlation_sum'] = df['x_z_acceleration_correlation'] + df['y_z_acceleration_correlation']

        if not inplace:
            return df
