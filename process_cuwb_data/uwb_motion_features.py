import functools
from operator import or_

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths
from sklearn import preprocessing

from .uwb_motion_filters import TrayMotionButterFiltFiltFilter, TrayMotionSavGolFilter
from .utils.log import logger
from .utils.util import filter_by_data_type


class FeatureExtraction:
    def __init__(
        self,
        frequency="100ms",
        position_filter=TrayMotionButterFiltFiltFilter(useSosFiltFilt=True),
        velocity_filter=TrayMotionSavGolFilter(),
    ):
        self.frequency = frequency
        self.position_filter = position_filter
        self.velocity_filter = velocity_filter

        self.le = preprocessing.LabelEncoder()
        self.le.fit(["dwtag100", "pt202"])

    VELOCITY_COLUMNS = [
        "quality",
        "anchor_count",
        # "x_position",
        # "y_position",
        # "z_position",
        "x_position_smoothed",
        "y_position_smoothed",
        "z_position_smoothed",
        "x_velocity_smoothed",
        "y_velocity_smoothed",
        "z_velocity_smoothed",
        "x_velocity_smoothed_magnitude",
        "y_velocity_smoothed_magnitude",
        "z_velocity_smoothed_magnitude",
        "velocity_vector_magnitude",
        "velocity_vector_magnitude_xy",
        "x_velocity_mean",
        "y_velocity_mean",
        "z_velocity_mean",
        "velocity_average_mean",
        "velocity_vector_magnitude_mean",
        "velocity_vector_magnitude_mean_xy",
        "x_velocity_stddev",
        "y_velocity_stddev",
        "z_velocity_stddev",
        "velocity_average_stddev",
        "velocity_vector_magnitude_stddev",
        "velocity_vector_magnitude_stddev_xy",
        "x_velocity_skew",
        "y_velocity_skew",
        "z_velocity_skew",
        "velocity_average_skew",
        "velocity_vector_magnitude_skew",
        "velocity_vector_magnitude_skew_xy",
        "x_velocity_variance",
        "y_velocity_variance",
        "z_velocity_variance",
        "velocity_average_variance",
        "velocity_vector_magnitude_variance",
        "velocity_vector_magnitude_variance_xy",
        "x_velocity_kurtosis",
        "y_velocity_kurtosis",
        "z_velocity_kurtosis",
        "velocity_average_kurtosis",
        "velocity_vector_magnitude_kurtosis",
        "velocity_vector_magnitude_kurtosis_xy",
        "x_y_velocity_correlation",
        "x_z_velocity_correlation",
        "y_z_velocity_correlation",
        "x_velocity_correlation_sum",
        "y_velocity_correlation_sum",
        "z_velocity_correlation_sum",
    ]

    ACCELERATION_COLUMNS = [
        "x_acceleration_normalized",
        "y_acceleration_normalized",
        "z_acceleration_normalized",
        "acceleration_vector_magnitude",
        "x_acceleration_mean",
        "y_acceleration_mean",
        "z_acceleration_mean",
        "acceleration_average_mean",
        "acceleration_vector_magnitude_mean",
        "x_acceleration_sum",
        "y_acceleration_sum",
        "z_acceleration_sum",
        "acceleration_average_sum",
        "acceleration_vector_magnitude_sum",
        "x_acceleration_min",
        "y_acceleration_min",
        "z_acceleration_min",
        "acceleration_average_min",
        "acceleration_vector_magnitude_min",
        "x_acceleration_max",
        "y_acceleration_max",
        "z_acceleration_max",
        "acceleration_average_max",
        "acceleration_vector_magnitude_max",
        "x_acceleration_stddev",
        "y_acceleration_stddev",
        "z_acceleration_stddev",
        "acceleration_average_stddev",
        "acceleration_vector_magnitude_stddev",
        "x_acceleration_skew",
        "y_acceleration_skew",
        "z_acceleration_skew",
        "acceleration_average_skew",
        "acceleration_vector_magnitude_skew",
        "x_acceleration_variance",
        "y_acceleration_variance",
        "z_acceleration_variance",
        "acceleration_average_variance",
        "acceleration_vector_magnitude_variance",
        "x_acceleration_kurtosis",
        "y_acceleration_kurtosis",
        "z_acceleration_kurtosis",
        "acceleration_average_kurtosis",
        "acceleration_vector_magnitude_kurtosis",
        "x_acceleration_energy",
        "y_acceleration_energy",
        "z_acceleration_energy",
        "acceleration_average_energy",
        "acceleration_vector_magnitude_energy",
        "x_y_acceleration_correlation",
        "x_z_acceleration_correlation",
        "y_z_acceleration_correlation",
        "x_acceleration_correlation_sum",
        "y_acceleration_correlation_sum",
        "z_acceleration_correlation_sum",
    ]

    GYROSCOPE_COLUMNS = ["x_dps", "y_dps", "z_dps"]

    MAGNETOMETER_COLUMNS = ["x_μT", "y_μT", "z_μT"]

    ALL_FEATURE_COLUMNS = [
        "device_part_number_label_id",
        *VELOCITY_COLUMNS,
        *ACCELERATION_COLUMNS,
        *GYROSCOPE_COLUMNS,
        *MAGNETOMETER_COLUMNS,
    ]

    def extract_motion_features_for_multiple_devices(
        self,
        df_uwb_data: pd.DataFrame,
        entity_type="all",
        fillna=None,
        join="outer",
    ):
        if (
            (df_uwb_data is None)
            or ((df_uwb_data is not None and len(df_uwb_data) == 0))
            or ((df_uwb_data is not None and "entity_type" not in df_uwb_data.columns))
        ):
            return None

        df_uwb_data = df_uwb_data.copy()

        if entity_type is None:
            entity_type = "all"

        if entity_type.lower() != "all":
            df_uwb_data = df_uwb_data[df_uwb_data["entity_type"].str.lower() == entity_type.lower()]

        df_uwb_data = self.filter_uwb_data_by_acceleration_activity(df_uwb_data=df_uwb_data)

        all_features = []
        for device_id, df_device_uwb_data in df_uwb_data.groupby(by="device_id"):
            logger.info(f"Calculating motion features for device ID {device_id}")

            df_device_uwb_data = df_device_uwb_data.sort_index()

            df_device_position = filter_by_data_type(df_device_uwb_data, "position")
            df_device_acceleration = filter_by_data_type(df_device_uwb_data, "accelerometer")
            df_device_gyroscope = filter_by_data_type(df_device_uwb_data, "gyroscope")
            df_device_magnetometer = filter_by_data_type(df_device_uwb_data, "magnetometer")

            df_features = self.extract_motion_features(
                df_position=df_device_position,
                df_acceleration=df_device_acceleration,
                df_gyroscope=df_device_gyroscope,
                df_magnetometer=df_device_magnetometer,
                fillna=fillna,
                join=join,
            )
            df_features["device_id"] = device_id

            device_part_number = df_device_uwb_data["device_part_number"].unique()[0].lower()
            df_features["device_part_number_label_id"] = self.le.transform([device_part_number])[0]

            all_features.append(df_features)

        df_all = pd.concat(all_features)
        return df_all

    def extract_tray_motion_features_for_multiple_devices(self, df_uwb_data):
        return self.extract_motion_features_for_multiple_devices(
            df_uwb_data=df_uwb_data,
            entity_type="tray",
        )

    def extract_motion_features(
        self,
        df_position,
        df_acceleration,
        df_gyroscope=None,
        df_magnetometer=None,
        fillna="forward_backward",
        join="outer",
    ):
        df_velocity_features = pd.DataFrame(columns=FeatureExtraction.VELOCITY_COLUMNS)
        if df_position is not None:
            df_velocity_features = self.extract_velocity_features(df=df_position)

        df_acceleration_features = pd.DataFrame(columns=FeatureExtraction.ACCELERATION_COLUMNS)
        if df_acceleration is not None:
            df_acceleration_features = self.extract_acceleration_features(df=df_acceleration)

        df_gyroscope_features = pd.DataFrame(columns=FeatureExtraction.GYROSCOPE_COLUMNS)
        if df_gyroscope is not None:
            df_gyroscope_features = self.extract_gyroscope_features(df=df_gyroscope)

        df_magnetometer_features = pd.DataFrame(columns=FeatureExtraction.MAGNETOMETER_COLUMNS)
        if df_magnetometer is not None:
            df_magnetometer_features = self.extract_magnetometer_features(df=df_magnetometer)

        df_features = (
            df_velocity_features.join(df_acceleration_features, how=join)
            .join(df_gyroscope_features, how=join)
            .join(df_magnetometer_features, how=join)
        )

        df_features = df_features.reindex(
            columns=[
                *FeatureExtraction.VELOCITY_COLUMNS,
                *FeatureExtraction.ACCELERATION_COLUMNS,
                *FeatureExtraction.GYROSCOPE_COLUMNS,
                *FeatureExtraction.MAGNETOMETER_COLUMNS,
            ]
        )

        df_features.replace([np.inf, -np.inf], np.nan, inplace=True)

        logger.info(f"Using fillna method: {fillna}")
        if fillna == "average":
            df_features.fillna(df_features.mean(), inplace=True)
        elif fillna == "drop":
            df_features.dropna(inplace=True)
        elif fillna == "pad":
            df_features.fillna(method="pad", inplace=True)
        elif fillna == "forward_backward":
            logger.info("Applying forward_backward filter to feature set")
            df_features = df_features.fillna(method="ffill").fillna(method="bfill")
        elif fillna == "interpolate":  # linear interpolation, use bfill to fill in nan's at front of dataframe
            logger.info("Applying standard bfill interpolation filter to feature set")
            df_features = df_features.interpolate().fillna(method="bfill")

        return df_features

    def extract_velocity_features(self, df):
        df = df.copy()

        if "x" in df.columns:
            df.rename(columns={"x": "x_position", "y": "y_position", "z": "z_position"}, inplace=True)

        if "x_meters" in df.columns:
            df.rename(
                columns={"x_meters": "x_position", "y_meters": "y_position", "z_meters": "z_position"}, inplace=True
            )

        df = self.average_xyz_duplicates(df, x_col="x_position", y_col="y_position", z_col="z_position")

        df = df.reindex(columns=["x_position", "y_position", "z_position", "quality", "anchor_count"])
        df = self.regularize_index_and_smooth(df)
        df = self.calculate_velocity_features(df=df)
        df = df.sort_index()
        return df

    def extract_acceleration_features(self, df):
        df = df.copy()
        if "x" in df.columns:
            df.rename(columns={"x": "x_gs", "y": "y_gs", "z": "z_gs"}, inplace=True)

        df = self.average_xyz_duplicates(df, x_col="x_gs", y_col="y_gs", z_col="z_gs")
        df = self.normalize_acceleration(df, x_col="x_gs", y_col="y_gs", z_col="z_gs")

        # df_acceleration_for_device_by_peaks = self.remove_wos_acceleration_peaks(
        #     df=df,
        #     x_col="x_acceleration_normalized",
        #     y_col="y_acceleration_normalized",
        #     z_col="z_acceleration_normalized",
        #     require_peak_across_all_axes=False)

        df = self.remove_wos_initial_acceleration_reading_by_time_gaps(df)

        df = df.reindex(columns=["x_gs", "y_gs", "z_gs"])
        df = self.regularize_index_and_smooth(df)
        df = self.calculate_acceleration_features(
            df=df,
        )
        df = df.drop(columns=["x_gs", "y_gs", "z_gs"]).sort_index()
        return df

    def extract_gyroscope_features(self, df):
        df = df.copy()

        if "x" in df.columns:
            df.rename(columns={"x": "x_dps", "y": "y_dps", "z": "z_dps"}, inplace=True)

        df = self.average_xyz_duplicates(df, x_col="x_dps", y_col="y_dps", z_col="z_dps")

        df = df.reindex(columns=["x_dps", "y_dps", "z_dps"])
        df = self.regularize_index_and_smooth(df)
        return df

    def extract_magnetometer_features(self, df):
        df = df.copy()

        if "x" in df.columns:
            df.rename(columns={"x": "x_μT", "y": "y_μT", "z": "z_μT"}, inplace=True)

        df = self.average_xyz_duplicates(df, x_col="x_μT", y_col="y_μT", z_col="z_μT")

        df = df.reindex(columns=["x_μT", "y_μT", "z_μT"])
        df = self.regularize_index_and_smooth(df)
        return df

    def regularize_index_and_smooth(self, df):
        df = df.copy()

        if len(df) == 0:
            return df

        df = df.astype(float)
        df = df.loc[~df.index.duplicated()].copy()

        start = df.index.min().floor(self.frequency)
        end = df.index.max().ceil(self.frequency)

        regularized_index = pd.date_range(start=start, end=end, freq=self.frequency)
        df = df.reindex(df.index.union(regularized_index))
        df = df.interpolate(method="time", limit=5, limit_area="inside")

        # Drop all rows that have an NA value (excluding the anchor_count column)
        df = df.reindex(regularized_index).dropna(subset=df.columns.difference(["anchor_count"]))
        # if fillna == 'interpolate':
        #     df = df.interpolate(method="linear", limit_area="inside")
        # else:
        #     df = df.fillna(value=0.0)  # .dropna(subset=df.columns.difference(["anchor_count"]))

        # df = df.reindex(regularized_index)
        df = df.bfill().ffill()
        return df

    def detect_peaks(self, np_array, width=None, min_height_as_percentage_of_max=0.8):
        """
        This method using Scipy find_peaks to extract peaks, peak 'widths', and peak 'prominences'.

        Note, data should be normalized before using this method.

        :param np_array: Expects a numpy array of values. This method works on a single axis of data at a time.
        :param width: See the Scipy find_peaks::width attribute. In short, this method defines the min/max width of a peak
        :param min_height_as_percentage_of_max:  See the Scipy find_peaks::height attribute. This method sets the minimum height based on a percentage of the max value in the provided numpy array.
        :return: (peaks, widths, prominences)
        """
        np_array = np_array.copy()
        np_array.reset_index(drop=True, inplace=True)

        # find_peaks will miss peaks at the front and back of an array unless we prepend and append min values to the array being analyzed
        augmented_np_array = np.concatenate(([min(np_array)], np_array, [min(np_array)]))

        peaks, _ = find_peaks(
            augmented_np_array, width=width, height=(max(augmented_np_array) * min_height_as_percentage_of_max, None)
        )
        widths = peak_widths(augmented_np_array, peaks)[0]
        prominences = peak_prominences(augmented_np_array, peaks)[0]

        # compensate for the prepended value in the augmented_x array that was analyzed
        peaks -= 1

        return peaks, widths, prominences

    def average_xyz_duplicates(self, df, x_col="x", y_col="y", z_col="z", group_by_cols=None, inplace=False):
        """
        Ciholas' IMU system will sometimes output multiple readings at the same time ("same time" after we lost access to the Ciholas network_time attribute")

        This method will average the x/y/z values for any of those duplicates and the duplicates will be removed.

        :param df: DF with time index
        :param x_col:
        :param y_col:
        :param z_col:
        :params group_by_cols: By default, the dataframe will be deduped by the index. Alternatively, an array of columns can be provided to group by.
        :param inplace:
        :return: Dataframe
        """
        if not inplace:
            df = df.copy()

        # ['socket_read_time', 'device_id']
        if group_by_cols is None:
            df["tmp_group_by_index"] = df.index
            group_by_cols = ["tmp_group_by_index"]

        df_averaged = df.groupby(group_by_cols).agg({x_col: np.mean, y_col: np.mean, z_col: np.mean})

        df = (
            df.drop_duplicates(group_by_cols)
            .drop(columns=[x_col, y_col, z_col])
            .merge(df_averaged, left_on=group_by_cols, right_index=True, how="left", suffixes=("_x", None))
        )

        if "tmp_group_by_index" in group_by_cols:
            df = df.drop(columns=["tmp_group_by_index"])

        return df

    def calculate_velocity_features(self, df, inplace=False):
        if not inplace:
            df = df.copy()

        if "x" in df.columns:
            df.rename(columns={"x": "x_position", "y": "y_position", "z": "z_position"}, inplace=True)

        df["x_position_smoothed"] = self.position_filter.filter(series=df["x_position"])
        df["y_position_smoothed"] = self.position_filter.filter(series=df["y_position"])
        df["z_position_smoothed"] = self.position_filter.filter(series=df["z_position"])
        # Old method of computing velocity, switched to savgol with deriv=1
        # df['x_velocity_smoothed']=df['x_acceleration_smoothed'].diff().divide(df.index.to_series().diff().apply(lambda dt: dt.total_seconds()))
        # df['y_velocity_smoothed']=df['y_acceleration_smoothed'].diff().divide(df.index.to_series().diff().apply(lambda dt: dt.total_seconds()))

        df["x_velocity_smoothed"] = self.velocity_filter.filter(df["x_position_smoothed"], deriv=1)
        df["y_velocity_smoothed"] = self.velocity_filter.filter(df["y_position_smoothed"], deriv=1)
        df["z_velocity_smoothed"] = self.velocity_filter.filter(df["z_position_smoothed"], deriv=1)

        df["x_velocity_smoothed_magnitude"] = df["x_velocity_smoothed"].abs()
        df["y_velocity_smoothed_magnitude"] = df["y_velocity_smoothed"].abs()
        df["z_velocity_smoothed_magnitude"] = df["z_velocity_smoothed"].abs()

        df["velocity_vector_magnitude"] = (
            df[["x_velocity_smoothed_magnitude", "y_velocity_smoothed_magnitude", "z_velocity_smoothed_magnitude"]]
            .pow(2)
            .sum(axis=1)
            .pow(0.5)
        )

        df["velocity_vector_magnitude_xy"] = (
            df[["x_velocity_smoothed_magnitude", "y_velocity_smoothed_magnitude"]].pow(2).sum(axis=1).pow(0.5)
        )

        window = pd.tseries.frequencies.to_offset(
            "2s"
        )  # int(1 / (pd.tseries.frequencies.to_offset(self.frequency).nanos / 1000000000))

        df["x_velocity_mean"] = df["x_velocity_smoothed_magnitude"].rolling(window=window, center=True).mean()
        df["y_velocity_mean"] = df["y_velocity_smoothed_magnitude"].rolling(window=window, center=True).mean()
        df["z_velocity_mean"] = df["z_velocity_smoothed_magnitude"].rolling(window=window, center=True).mean()
        df["velocity_average_mean"] = df[["x_velocity_mean", "y_velocity_mean", "z_velocity_mean"]].mean(axis=1)

        df["velocity_vector_magnitude_mean"] = (
            df["velocity_vector_magnitude"].rolling(window=window, center=True).mean()
        )
        df["velocity_vector_magnitude_mean_xy"] = (
            df["velocity_vector_magnitude_xy"].rolling(window=window, center=True).mean()
        )

        df["x_velocity_stddev"] = df["x_velocity_smoothed_magnitude"].rolling(window=window, center=True).std()
        df["y_velocity_stddev"] = df["y_velocity_smoothed_magnitude"].rolling(window=window, center=True).std()
        df["z_velocity_stddev"] = df["z_velocity_smoothed_magnitude"].rolling(window=window, center=True).std()
        df["velocity_average_stddev"] = df[["x_velocity_stddev", "y_velocity_stddev", "z_velocity_stddev"]].mean(axis=1)

        df["velocity_vector_magnitude_stddev"] = (
            df["velocity_vector_magnitude"].rolling(window=window, center=True).std()
        )
        df["velocity_vector_magnitude_stddev_xy"] = (
            df["velocity_vector_magnitude_xy"].rolling(window=window, center=True).std()
        )

        df["x_velocity_skew"] = df["x_velocity_smoothed_magnitude"].rolling(window=window, center=True).skew()
        df["y_velocity_skew"] = df["y_velocity_smoothed_magnitude"].rolling(window=window, center=True).skew()
        df["z_velocity_skew"] = df["z_velocity_smoothed_magnitude"].rolling(window=window, center=True).skew()
        df["velocity_average_skew"] = df[["x_velocity_skew", "y_velocity_skew", "z_velocity_skew"]].mean(axis=1)

        df["velocity_vector_magnitude_skew"] = (
            df["velocity_vector_magnitude"].rolling(window=window, center=True).skew()
        )
        df["velocity_vector_magnitude_skew_xy"] = (
            df["velocity_vector_magnitude_xy"].rolling(window=window, center=True).skew()
        )

        df["x_velocity_variance"] = df["x_velocity_smoothed_magnitude"].rolling(window=window, center=True).var()
        df["y_velocity_variance"] = df["y_velocity_smoothed_magnitude"].rolling(window=window, center=True).var()
        df["z_velocity_variance"] = df["z_velocity_smoothed_magnitude"].rolling(window=window, center=True).var()
        df["velocity_average_variance"] = df[
            ["x_velocity_variance", "y_velocity_variance", "z_velocity_variance"]
        ].mean(axis=1)

        df["velocity_vector_magnitude_variance"] = (
            df["velocity_vector_magnitude"].rolling(window=window, center=True).var()
        )
        df["velocity_vector_magnitude_variance_xy"] = (
            df["velocity_vector_magnitude_xy"].rolling(window=window, center=True).var()
        )

        df["x_velocity_kurtosis"] = df["x_velocity_smoothed_magnitude"].rolling(window=window, center=True).kurt()
        df["y_velocity_kurtosis"] = df["y_velocity_smoothed_magnitude"].rolling(window=window, center=True).kurt()
        df["z_velocity_kurtosis"] = df["z_velocity_smoothed_magnitude"].rolling(window=window, center=True).kurt()
        df["velocity_average_kurtosis"] = df[
            ["x_velocity_kurtosis", "y_velocity_kurtosis", "z_velocity_kurtosis"]
        ].mean(axis=1)

        df["velocity_vector_magnitude_kurtosis"] = (
            df["velocity_vector_magnitude"].rolling(window=window, center=True).kurt()
        )
        df["velocity_vector_magnitude_kurtosis_xy"] = (
            df["velocity_vector_magnitude_xy"].rolling(window=window, center=True).kurt()
        )

        df["x_y_velocity_correlation"] = (
            df["x_velocity_smoothed_magnitude"].rolling(window=window).corr(df["y_velocity_smoothed_magnitude"])
        )
        df["x_z_velocity_correlation"] = (
            df["x_velocity_smoothed_magnitude"].rolling(window=window).corr(df["z_velocity_smoothed_magnitude"])
        )
        df["y_z_velocity_correlation"] = (
            df["y_velocity_smoothed_magnitude"].rolling(window=window).corr(df["z_velocity_smoothed_magnitude"])
        )
        df["x_velocity_correlation_sum"] = df["x_y_velocity_correlation"] + df["x_z_velocity_correlation"]
        df["y_velocity_correlation_sum"] = df["x_y_velocity_correlation"] + df["y_z_velocity_correlation"]
        df["z_velocity_correlation_sum"] = df["x_z_velocity_correlation"] + df["y_z_velocity_correlation"]

        if not inplace:
            return df

    def filter_uwb_data_by_acceleration_activity(self, df_uwb_data: pd.DataFrame, inplace: bool = False):
        """
        Filters UWB data by determining the "active periods" of each sensor.
        Active periods are periods when acceleration data is being captured. Acceleration data
        is not captured when a device is sleeping due to Wake on Shake. Note, while sleeping,
        position data is reported/captured once a minute. This makes position data a
        little more difficult to work with, so we filter by the acceleration data's behavior.

        :param df_uwb_data: UWB data for all devices
        :param inplace: Whether to modify the provided df_uwb_data dataframe directly

        :return: A filtered df_uwb_data dataframe
        """
        if not inplace:
            df_uwb_data = df_uwb_data.copy()

        filtered_uwb_data = []
        for device_id, df_device_uwb_data in df_uwb_data.groupby(by="device_id"):
            logger.info(f"Filtering UWB data by active sessions based on acceleration data for device ID: {device_id}")

            df_device_accelerometer_data = df_device_uwb_data[df_device_uwb_data["type"] == "accelerometer"].copy()
            df_device_accelerometer_data["active_session_id"] = (
                df_device_accelerometer_data.index.to_series().diff() >= pd.to_timedelta("60 seconds")
            ).cumsum()

            active_session_start_end_times = []
            for _, df_device_accelerometer_session in df_device_accelerometer_data.groupby(by=["active_session_id"]):
                active_session_start_end_times.append(
                    {
                        "start": df_device_accelerometer_session.index.min(),
                        "end": df_device_accelerometer_session.index.max(),
                    }
                )
            df_active_session_start_end_times = pd.DataFrame(active_session_start_end_times)

            def _filter_by_start_end_times(start_end_times):
                return (df_device_uwb_data.index >= start_end_times["start"]) & (
                    df_device_uwb_data.index <= start_end_times["end"]
                )

            start_end_times_mask_list = df_active_session_start_end_times.apply(_filter_by_start_end_times, axis=1)
            if len(start_end_times_mask_list) > 0:
                start_end_times_or_mask = functools.reduce(or_, start_end_times_mask_list)
                filtered_uwb_data.append(df_device_uwb_data[start_end_times_or_mask])

        return pd.concat(filtered_uwb_data)

    def remove_wos_acceleration_peaks(
        self, df, x_col="x", y_col="y", z_col="z", require_peak_across_all_axes=False, inplace=False
    ):
        """
        Method was created to correct for erroneous acceleration values introduced by WoS. Wake on Shake (WoS) has the tendency to spike when awaking. When analyzing the data, it appeared this data spike is usually contained to a single reading before the readings report sensible data.

        This method will remove all spikes from a provided Dataframe. Use the cols attributes to specify the x/y/z columns.

        Note, data should be normalized before using this method.

        :param df:
        :param x_col:
        :param y_col:
        :param z_col:
        :param require_peak_across_all_axes: When set to True the method will only remove a "peak candidate" if the peak occurred across all dimensions (x, y, and z)
        :param inplace:
        :return: Dataframe
        """
        if not inplace:
            df = df.copy()

        x_peaks, x_widths, _ = self.detect_peaks(df[x_col], width=1, min_height_as_percentage_of_max=0.6)
        y_peaks, y_widths, _ = self.detect_peaks(df[y_col], width=1, min_height_as_percentage_of_max=0.6)
        z_peaks, z_widths, _ = self.detect_peaks(df[z_col], width=1, min_height_as_percentage_of_max=0.6)

        all_peaks = np.concatenate((x_peaks, y_peaks, z_peaks))
        unique_peaks, peak_counts = np.unique(all_peaks, return_counts=True)

        peaks_idxs = unique_peaks
        if require_peak_across_all_axes:
            peaks_idxs = []
            for idx, count in enumerate(peak_counts):
                if count >= 3:
                    peaks_idxs.append(unique_peaks[idx])

        logger.info(f"Correcting for WoS, dropping {len(peaks_idxs)} indices by identifying acceleration peaks")

        df = df.drop(df.iloc[peaks_idxs].index)

        if not inplace:
            return df

    def remove_wos_initial_acceleration_reading_by_time_gaps(self, df_acceleration, inplace=False):
        """
        Method was created to correct for erroneous acceleration values introduced by WoS.

        Scans the provided dataframe for any time gaps (as defined by 0.5 seconds). Removes the first value following a time gap.
        :param inplace:
        :return:
        """
        if not inplace:
            df_acceleration = df_acceleration.copy()

        df_acceleration["gap"] = df_acceleration.sort_index().index.to_series().diff() > pd.to_timedelta("0.5 seconds")
        drop_index = df_acceleration.loc[df_acceleration["gap"]].index

        logger.info(
            f"Correcting for WoS, dropping {len(drop_index) + 1} indices by searching for time gaps and removing first data instance"
        )

        # Drop the first item that follows each "gap" in time, also drop the very first row which won't be identified with the diff() method
        df_acceleration = df_acceleration.drop(drop_index)

        if len(df_acceleration) > 0:
            df_acceleration = df_acceleration.drop(df_acceleration.index[0])

        if not inplace:
            return df_acceleration

    def normalize_acceleration(self, df, x_col="x", y_col="y", z_col="z", inplace=False):
        if not inplace:
            df = df.copy()

        df["x_acceleration_normalized"] = np.absolute(np.subtract(df[x_col], df[x_col].mean()))
        df["y_acceleration_normalized"] = np.absolute(np.subtract(df[y_col], df[y_col].mean()))
        df["z_acceleration_normalized"] = np.absolute(np.subtract(df[z_col], df[z_col].mean()))

        if not inplace:
            return df

    def calculate_acceleration_features(self, df, inplace=False):
        if not inplace:
            df = df.copy()

        if "x" in df.columns:
            df.rename(columns={"x": "x_gs", "y": "y_gs", "z": "z_gs"}, inplace=True)

        df = self.normalize_acceleration(df, x_col="x_gs", y_col="y_gs", z_col="z_gs")

        df["acceleration_vector_magnitude"] = (
            df[["x_acceleration_normalized", "y_acceleration_normalized", "z_acceleration_normalized"]]
            .pow(2)
            .sum(axis=1)
            .pow(0.5)
        )

        window = pd.tseries.frequencies.to_offset(
            "2s"
        )  # int(1 / (pd.tseries.frequencies.to_offset(self.frequency).nanos / 1000000000))

        df["x_acceleration_mean"] = df["x_acceleration_normalized"].rolling(window=window, center=True).mean()
        df["y_acceleration_mean"] = df["y_acceleration_normalized"].rolling(window=window, center=True).mean()
        df["z_acceleration_mean"] = df["z_acceleration_normalized"].rolling(window=window, center=True).mean()
        df["acceleration_average_mean"] = df[
            ["x_acceleration_mean", "y_acceleration_mean", "z_acceleration_mean"]
        ].mean(axis=1)

        df["acceleration_vector_magnitude_mean"] = (
            df["acceleration_vector_magnitude"].rolling(window=window, center=True).mean()
        )

        df["x_acceleration_sum"] = df["x_acceleration_normalized"].rolling(window=window, center=True).sum()
        df["y_acceleration_sum"] = df["y_acceleration_normalized"].rolling(window=window, center=True).sum()
        df["z_acceleration_sum"] = df["z_acceleration_normalized"].rolling(window=window, center=True).sum()
        df["acceleration_average_sum"] = df[["x_acceleration_sum", "y_acceleration_sum", "z_acceleration_sum"]].mean(
            axis=1
        )

        df["acceleration_vector_magnitude_sum"] = (
            df["acceleration_vector_magnitude"].rolling(window=window, center=True).sum()
        )

        df["x_acceleration_min"] = df["x_acceleration_normalized"].rolling(window=window, center=True).min()
        df["y_acceleration_min"] = df["y_acceleration_normalized"].rolling(window=window, center=True).min()
        df["z_acceleration_min"] = df["z_acceleration_normalized"].rolling(window=window, center=True).min()
        df["acceleration_average_min"] = df[["x_acceleration_min", "y_acceleration_min", "z_acceleration_min"]].mean(
            axis=1
        )

        df["acceleration_vector_magnitude_min"] = (
            df["acceleration_vector_magnitude"].rolling(window=window, center=True).min()
        )

        df["x_acceleration_max"] = df["x_acceleration_normalized"].rolling(window=window, center=True).max()
        df["y_acceleration_max"] = df["y_acceleration_normalized"].rolling(window=window, center=True).max()
        df["z_acceleration_max"] = df["z_acceleration_normalized"].rolling(window=window, center=True).max()
        df["acceleration_average_max"] = df[["x_acceleration_max", "y_acceleration_max", "z_acceleration_max"]].mean(
            axis=1
        )

        df["acceleration_vector_magnitude_max"] = (
            df["acceleration_vector_magnitude"].rolling(window=window, center=True).max()
        )

        df["x_acceleration_stddev"] = df["x_acceleration_normalized"].rolling(window=window, center=True).std()
        df["y_acceleration_stddev"] = df["y_acceleration_normalized"].rolling(window=window, center=True).std()
        df["z_acceleration_stddev"] = df["z_acceleration_normalized"].rolling(window=window, center=True).std()
        df["acceleration_average_stddev"] = df[
            ["x_acceleration_stddev", "y_acceleration_stddev", "z_acceleration_stddev"]
        ].mean(axis=1)

        df["acceleration_vector_magnitude_stddev"] = (
            df["acceleration_vector_magnitude"].rolling(window=window, center=True).std()
        )

        df["x_acceleration_skew"] = df["x_acceleration_normalized"].rolling(window=window, center=True).skew()
        df["y_acceleration_skew"] = df["y_acceleration_normalized"].rolling(window=window, center=True).skew()
        df["z_acceleration_skew"] = df["z_acceleration_normalized"].rolling(window=window, center=True).skew()
        df["acceleration_average_skew"] = df[
            ["x_acceleration_skew", "y_acceleration_skew", "z_acceleration_skew"]
        ].mean(axis=1)

        df["acceleration_vector_magnitude_skew"] = (
            df["acceleration_vector_magnitude"].rolling(window=window, center=True).skew()
        )

        df["x_acceleration_variance"] = df["x_acceleration_normalized"].rolling(window=window, center=True).var()
        df["y_acceleration_variance"] = df["y_acceleration_normalized"].rolling(window=window, center=True).var()
        df["z_acceleration_variance"] = df["z_acceleration_normalized"].rolling(window=window, center=True).var()
        df["acceleration_average_variance"] = df[
            ["x_acceleration_variance", "y_acceleration_variance", "z_acceleration_variance"]
        ].mean(axis=1)

        df["acceleration_vector_magnitude_variance"] = (
            df["acceleration_vector_magnitude"].rolling(window=window, center=True).var()
        )

        df["x_acceleration_kurtosis"] = df["x_acceleration_normalized"].rolling(window=window, center=True).kurt()
        df["y_acceleration_kurtosis"] = df["y_acceleration_normalized"].rolling(window=window, center=True).kurt()
        df["z_acceleration_kurtosis"] = df["z_acceleration_normalized"].rolling(window=window, center=True).kurt()
        df["acceleration_average_kurtosis"] = df[
            ["x_acceleration_kurtosis", "y_acceleration_kurtosis", "z_acceleration_kurtosis"]
        ].mean(axis=1)

        df["acceleration_vector_magnitude_kurtosis"] = (
            df["acceleration_vector_magnitude"].rolling(window=window, center=True).kurt()
        )

        df["x_acceleration_energy"] = df["x_acceleration_normalized"].pow(2).rolling(window=window, center=True).mean()
        df["y_acceleration_energy"] = df["y_acceleration_normalized"].pow(2).rolling(window=window, center=True).mean()
        df["z_acceleration_energy"] = df["z_acceleration_normalized"].pow(2).rolling(window=window, center=True).mean()
        df["acceleration_average_energy"] = df[
            ["x_acceleration_energy", "y_acceleration_energy", "z_acceleration_energy"]
        ].mean(axis=1)

        df["acceleration_vector_magnitude_energy"] = (
            df["acceleration_vector_magnitude"].pow(2).rolling(window=window, center=True).mean()
        )

        df["x_y_acceleration_correlation"] = (
            df["x_acceleration_normalized"].rolling(window=window).corr(df["y_acceleration_normalized"])
        )
        df["x_z_acceleration_correlation"] = (
            df["x_acceleration_normalized"].rolling(window=window).corr(df["z_acceleration_normalized"])
        )
        df["y_z_acceleration_correlation"] = (
            df["y_acceleration_normalized"].rolling(window=window).corr(df["z_acceleration_normalized"])
        )
        df["x_acceleration_correlation_sum"] = df["x_y_acceleration_correlation"] + df["x_z_acceleration_correlation"]
        df["y_acceleration_correlation_sum"] = df["x_y_acceleration_correlation"] + df["y_z_acceleration_correlation"]
        df["z_acceleration_correlation_sum"] = df["x_z_acceleration_correlation"] + df["y_z_acceleration_correlation"]

        if not inplace:
            return df
