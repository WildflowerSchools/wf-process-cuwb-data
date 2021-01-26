import numpy as np
import pandas as pd

from .io import load_csv
from .log import logger
from .honeycomb import fetch_environment_by_name, fetch_material_tray_devices_assignments, fetch_raw_cuwb_data
from .uwb_extract_data import extract_by_data_type_and_format, extract_by_entity_type
from .uwb_motion_classifiers import TrayCarryClassifier
from .uwb_motion_events import extract_carry_events_by_device
from .uwb_motion_features import FeatureExtraction
from .uwb_motion_ground_truth import combine_features_with_ground_truth_data, validate_ground_truth
from .uwb_motion_interactions import extract_tray_device_interactions, validate_tray_centroids_dataframe


def fetch_tray_device_assignments(
        environment_name,
        start_time,
        end_time):
    environment = fetch_environment_by_name(environment_name=environment_name)
    if environment is None:
        return None

    environment_id = environment['environment_id']
    df_tray_device_assignments = fetch_material_tray_devices_assignments(environment_id, start_time, end_time)
    return df_tray_device_assignments


def fetch_cuwb_data(
    environment_name,
    start_time,
    end_time,
    entity_type='all',
    data_type='raw',
    read_chunk_size=2,
    device_type='UWBTAG',
    environment_assignment_info=False,
    entity_assignment_info=False
):
    if entity_type not in ['tray', 'person', 'all']:
        raise Exception("Invalid 'entity_type' value: {}".format(type))

    if data_type not in ['position', 'accelerometer', 'status', 'raw']:
        raise Exception("Invalid 'data_type' value: {}".format(type))

    df = fetch_raw_cuwb_data(
        environment_name,
        start_time,
        end_time,
        read_chunk_size,
        device_type,
        environment_assignment_info,
        entity_assignment_info
    )

    df = extract_by_entity_type(df, entity_type)
    return extract_by_data_type_and_format(df, data_type)


def fetch_motion_features(environment, start, end, entity_type='all', include_meta_fields=False, fillna=None):
    df = fetch_cuwb_data(environment,
                         start,
                         end,
                         entity_type=entity_type,
                         data_type='raw',
                         environment_assignment_info=True,
                         entity_assignment_info=True)

    df_features = extract_motion_features(
        df_position=extract_by_data_type_and_format(df, data_type='position'),
        df_acceleration=extract_by_data_type_and_format(df, data_type='accelerometer'),
        entity_type=entity_type,
        fillna=fillna
    )

    # TODO: Resolve assumption that each device is only assigned to a single material
    # That assumption means if a tray is reassigned to a new material, the join will
    # create duplicates records
    if include_meta_fields and len(df) > 0:
        df_meta_fields = df[[
            'device_id', 'device_name', 'device_tag_id',
            'device_mac_address', 'device_part_number', 'device_serial_number', 'entity_type',
            'person_id', 'person_name', 'person_short_name', 'tray_id',
            'tray_name', 'material_assignment_id', 'material_id', 'material_name']
        ].set_index('device_id').drop_duplicates()

        duplicate_error = False
        for device_id, count in df_meta_fields.index.value_counts().items():
            if count > 1:
                duplicate_error = True
                logger.error(
                    "Unexpected duplicate device_id - '{}' when fetching CUWB data. This may be caused because a tray device had been assigned to multiple materials during given time period".format(device_id))

        if duplicate_error:
            return None

        return df_features.join(df_meta_fields, on='device_id', how='left')
    else:
        return df_features


def extract_motion_features(df_position, df_acceleration, entity_type='all', fillna=None):
    f = FeatureExtraction()
    return f.extract_motion_features_for_multiple_devices(df_position, df_acceleration, entity_type, fillna=fillna)


def generate_tray_carry_groundtruth(groundtruth_csv):
    try:
        df_groundtruth = load_csv(groundtruth_csv)
        valid, msg = validate_ground_truth(df_groundtruth)
        if not valid:
            logger.error(msg)
            return None
    except Exception as err:
        logger.error(err)
        return None

    df_features = None
    for (environment, start_datetime), group_df in df_groundtruth.groupby(
            by=['environment', pd.Grouper(key='start_datetime', freq='D')]):
        start = group_df.agg({'start_datetime': [np.min]}).iloc[0]['start_datetime']
        end = group_df.agg({'end_datetime': [np.max]}).iloc[0]['end_datetime']

        # Until fetch_motion_features can return data safely within exact
        # bounds, add 60 minutes offsets to start and end
        df_environment_features = fetch_motion_features(environment=environment,
                                                        start=(start - pd.DateOffset(minutes=60)),
                                                        end=(end + pd.DateOffset(minutes=60)),
                                                        entity_type='tray')

        if df_features is None:
            df_features = df_environment_features.copy()
        else:
            df_features = df_features.append(df_environment_features)

    try:
        df_groundtruth_features = combine_features_with_ground_truth_data(df_features, df_groundtruth)
    except Exception as err:
        logger.error(err)
        return None

    logger.info("Tray Carry groundtruth features breakdown by device\n{}".format(
        df_groundtruth_features.fillna('NA').groupby(['device_id', 'ground_truth_state']).size()))

    return df_groundtruth_features


def generate_tray_carry_model(groundtruth_features, tune=False):
    tc = TrayCarryClassifier()
    if tune:
        tc.tune(df_groundtruth=groundtruth_features)
        return None
    else:
        return tc.train(df_groundtruth=groundtruth_features, scale_features=False)


def infer_tray_carry(model, scaler, df_tray_features):
    """
    Classifies each moment of features dataframe into a carried or not carried state

    :param model: Tray carry classifier (RandomForest Model)
    :param scaler: Tray carry scaling model used to standardize features
    :param df_tray_features: Dataframe with uwb data containing uwb_motion_classifiers.DEFAULT_FEATURE_FIELD_NAMES
    :return: Dataframe with uwb data containing a "predicted_state" column
    """
    tc = TrayCarryClassifier(model=model, feature_scaler=scaler)
    return tc.inference(df_tray_features)


def extract_tray_carry_events_from_inferred(df_inferred):
    """
    Extract carry events from inferred carried or not carried states (see infer_tray_carry)

    :param df_inferred: Dataframe with uwb data containing a "predicted_state" column
    :return: Dataframe containing carry events (device_id (tray ID), start, end)
    """
    return extract_carry_events_by_device(df_inferred)


def extract_tray_interactions(df_features, df_carry_events, tray_positions_csv=None):
    """
    Extract carry interactions (person, tray, carry_event - FROM_SHELF/TO_SHELF/etc) from carried events (see extract_tray_carry_events_from_inferred)

    :param df_carry_events: Dataframe with carry events (device_id, start, end)
    :return: Dataframe containing carry interactions (person_id, device_id, start, end, carry_event)
    """

    df_tray_centroids = None
    if tray_positions_csv is not None:
        try:
            df_tray_centroids = load_csv(tray_positions_csv)
            valid, msg = validate_tray_centroids_dataframe(df_tray_centroids)
            if not valid:
                logger.error(msg)
                return None
        except Exception as err:
            logger.error(err)
            return None

    return extract_tray_device_interactions(df_features, df_carry_events, df_tray_centroids)
