from process_cuwb_data.uwb_extract_data import extract_by_data_type_and_format
from process_cuwb_data.uwb_motion_features import FeatureExtraction

import numpy as np
import pandas as pd


class TestUWBMotionFeatures:
    @classmethod
    def prep_test_cuwb_data(cls, cuwb_dataframe):
        # Build dataframe with:
        #   1 tray device that has both position and acceleration
        #   1 person device that has both position and acceleration
        #   1 person device that has only position
        test_device_ids = []

        has_tray_with_position_and_acceleration = None
        has_person_with_position_and_acceleration = None
        has_person_with_position_only = None
        for device_id in pd.unique(cuwb_dataframe['device_id']):
            device_position_filter = ((cuwb_dataframe['device_id'] == device_id) & (cuwb_dataframe['type'] == 'position'))
            device_accelerometer_filter = ((cuwb_dataframe['device_id'] == device_id) & (cuwb_dataframe['type'] == 'accelerometer'))

            if has_tray_with_position_and_acceleration is None:
                if (len(cuwb_dataframe[device_position_filter]) > 0 and
                        len(cuwb_dataframe[device_accelerometer_filter]) > 0 and
                        cuwb_dataframe[device_position_filter]['entity_type'][0] == 'Tray'):
                    test_device_ids.append(device_id)
                    has_tray_with_position_and_acceleration = device_id
                    continue

            if has_person_with_position_and_acceleration is None:
                if (len(cuwb_dataframe[device_position_filter]) > 0 and
                        len(cuwb_dataframe[device_accelerometer_filter]) > 0 and
                        cuwb_dataframe[device_position_filter]['entity_type'][0] == 'Person'):
                    test_device_ids.append(device_id)
                    has_person_with_position_and_acceleration = device_id
                    continue

            if has_person_with_position_only is None:
                if (len(cuwb_dataframe[device_position_filter]) > 0 and
                        len(cuwb_dataframe[device_accelerometer_filter]) == 0 and
                        cuwb_dataframe[device_position_filter]['entity_type'][0] == 'Person'):
                    test_device_ids.append(device_id)
                    has_person_with_position_only = device_id
                    continue

        assert has_tray_with_position_and_acceleration is not None, "Expected tray device with position and acceleration data"
        assert has_person_with_position_and_acceleration is not None, "Expected person device with position and acceleration data"
        assert has_person_with_position_only is not None, "Expected person device with position data only"

        return cuwb_dataframe[cuwb_dataframe['device_id'].isin(test_device_ids)]

    def test_extract_motion_features_handles_missing_accelerometer_data(self, cuwb_dataframe):
        df_test_cuwb_data = TestUWBMotionFeatures.prep_test_cuwb_data(cuwb_dataframe)

        f = FeatureExtraction()
        df_motion_features = f.extract_motion_features_for_multiple_devices(
            df_position=extract_by_data_type_and_format(df_test_cuwb_data, data_type='position'),
            df_acceleration=extract_by_data_type_and_format(df_test_cuwb_data, data_type='accelerometer'),
            entity_type='all'
        )

        count_unique_devices_original = len(pd.unique(df_test_cuwb_data['device_id']))
        count_unique_devices_motion_data = len(pd.unique(df_motion_features['device_id']))

        assert count_unique_devices_original > 0, "Expected test to contain at least one device"
        assert count_unique_devices_motion_data == count_unique_devices_original, "Expected device in to equal devices out"

    def test_extract_motion_features_extract_single_motion_type(self, cuwb_dataframe):
        df_test_cuwb_data = TestUWBMotionFeatures.prep_test_cuwb_data(cuwb_dataframe)

        f = FeatureExtraction()
        df_motion_features = f.extract_motion_features_for_multiple_devices(
            df_position=extract_by_data_type_and_format(df_test_cuwb_data, data_type='position'),
            entity_type='all'
        )

        assert len(pd.unique(df_motion_features['device_id'])) == 3
        assert len(pd.unique(df_motion_features['x_acceleration_normalized'])) == 1
        assert np.isnan(pd.unique(df_motion_features['x_acceleration_normalized']))

        df_motion_features = f.extract_motion_features_for_multiple_devices(
            df_acceleration=extract_by_data_type_and_format(df_test_cuwb_data, data_type='accelerometer'),
            entity_type='all'
        )

        assert len(pd.unique(df_motion_features['device_id'])) == 2
        assert len(pd.unique(df_motion_features['x_velocity_smoothed_magnitude'])) == 1
        assert np.isnan(pd.unique(df_motion_features['x_velocity_smoothed_magnitude']))

    def test_extract_motion_features_extract_by_entity_type(self, cuwb_dataframe):
        df_test_cuwb_data = TestUWBMotionFeatures.prep_test_cuwb_data(cuwb_dataframe)

        f = FeatureExtraction()
        df_motion_features = f.extract_motion_features_for_multiple_devices(
            df_position=extract_by_data_type_and_format(df_test_cuwb_data, data_type='position'),
            df_acceleration=extract_by_data_type_and_format(df_test_cuwb_data, data_type='accelerometer'),
            entity_type='tray'
        )

        assert len(pd.unique(df_test_cuwb_data.loc[df_test_cuwb_data['entity_type'].str.lower() == 'tray']['device_id'])) == 1
        assert len(pd.unique(df_motion_features['device_id'])) == 1

        df_motion_features = f.extract_motion_features_for_multiple_devices(
            df_position=extract_by_data_type_and_format(df_test_cuwb_data, data_type='position'),
            df_acceleration=extract_by_data_type_and_format(df_test_cuwb_data, data_type='accelerometer'),
            entity_type='person'
        )
        assert len(pd.unique(df_test_cuwb_data.loc[df_test_cuwb_data['entity_type'].str.lower() == 'person']['device_id'])) == 2
        assert len(pd.unique(df_motion_features['device_id'])) == 2
