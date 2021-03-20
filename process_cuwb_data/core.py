from honeycomb_io import fetch_cuwb_position_data, fetch_cuwb_accelerometer_data, add_device_assignment_info, add_device_entity_assignment_info, add_tray_material_assignment_info, fetch_environment_by_name, fetch_material_tray_devices_assignments, fetch_raw_cuwb_data
import numpy as np
import pandas as pd

from .utils.io import load_csv
from .utils.log import logger
from .uwb_extract_data import extract_by_data_type_and_format, extract_by_entity_type
from .uwb_motion_classifier_human_activity import HumanActivityClassifier
from .uwb_motion_classifier_tray_carry import TrayCarryClassifier
from .uwb_motion_events import extract_carry_events_by_device
from .uwb_motion_features import FeatureExtraction
import process_cuwb_data.uwb_motion_ground_truth as ground_truth
from .uwb_motion_interactions import extract_tray_device_interactions
from .uwb_predict_tray_centroids import classifier_filter_no_movement_from_tray_features, predict_tray_centroids


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
    device_type='UWBTAG',
    environment_assignment_info=False,
    entity_assignment_info=False
):
    if entity_type not in ['tray', 'person', 'all']:
        raise Exception("Invalid 'entity_type' value: {}".format(type))

    if data_type not in ['position', 'accelerometer', 'status', 'raw']:
        raise Exception("Invalid 'data_type' value: {}".format(type))

    df = fetch_raw_cuwb_data(
        environment_name=environment_name,
        start_time=start_time,
        end_time=end_time,
        device_type=device_type,
        environment_assignment_info=environment_assignment_info,
        entity_assignment_info=entity_assignment_info
    )

    df = extract_by_entity_type(df, entity_type)
    return extract_by_data_type_and_format(df, data_type)

def fetch_motion_features_new(
    environment,
    start,
    end,
    entity_type='all',
    include_meta_fields=False,
    fillna=None
):
    # Fetch position data
    df_position = fetch_cuwb_position_data(
        start=start,
        end=end,
        device_ids=None,
        environment_id=None,
        environment_name=environment,
        device_types=['UWBTAG'],
        output_format='dataframe'
    )
    # Add metadata
    df_position = add_device_assignment_info(df_position)
    df_position = add_device_entity_assignment_info(df_position)
    df_position = add_tray_material_assignment_info(df_position)
    # Filter on entity type
    df_position = filter_entity_type(df_position, entity_type=entity_type)
    # Reorganize columns as expected by extract_motion_features()
    df_position['type'] = 'position'
    df_position.rename(
        columns={
            'x': 'x_meters',
            'y': 'y_meters',
            'z': 'z_meters'
        },
        inplace=True
    )
    df_position.reset_index(drop=True, inplace=True)
    df_position.set_index('timestamp', inplace=True)
    # Fetch acceleration data
    df_acceleration = fetch_cuwb_accelerometer_data(
        start=start,
        end=end,
        device_ids=None,
        environment_id=None,
        environment_name=environment,
        device_types=['UWBTAG'],
        output_format='dataframe'
    )
    # Add metadata
    df_acceleration = add_device_assignment_info(df_acceleration)
    df_acceleration = add_device_entity_assignment_info(df_acceleration)
    df_acceleration = add_tray_material_assignment_info(df_acceleration)
    # Filter on entity type
    df_acceleration = filter_entity_type(df_acceleration, entity_type=entity_type)
    # Reorganize columns as expected by extract_motion_features()
    df_acceleration['type'] = 'accelerometer'
    df_acceleration.rename(
        columns={
        'x': 'x_gs',
        'y': 'y_gs',
        'z': 'z_gs'
        },
        inplace=True
    )
    df_acceleration.reset_index(drop=True, inplace=True)
    df_acceleration.set_index('timestamp', inplace=True)
    # Extract motion features
    df_motion_features = extract_motion_features(
        df_position=df_position,
        df_acceleration=df_acceleration,
        entity_type=entity_type,
        fillna=fillna
    )
    # Add metadata fields if requested
    if include_meta_fields and (len(df_position) > 0 or len(df_acceleration) > 0):
        df_all_datatypes = pd.concat((df_position, df_acceleration))
        df_meta_fields = (
            df_all_datatypes.loc[:, [
                'device_id',
                'device_name',
                'device_tag_id',
                'device_mac_address',
                'device_part_number',
                'device_serial_number',
                'entity_type',
                'person_id',
                'person_name',
                'person_short_name',
                'tray_id',
                'tray_name',
                'material_assignment_id',
                'material_id',
                'material_name'
            ]]
            .set_index('device_id')
            .drop_duplicates()
            .copy()
        )
        # We don't need to check for duplicate device IDs because our functions
        # for fetching device assignments, device entity assignments, and tray
        # material assignments all enforce uniqueness by default
        return df_motion_features.join(df_meta_fields, on='device_id', how='left')
    else:
        return df_motion_features

def filter_entity_type(dataframe, entity_type='all'):
    if entity_type == 'all':
        return dataframe
    elif entity_type == 'tray':
        return dataframe.loc[dataframe['entity_type'] == 'Tray'].copy()
    elif entity_type == 'person':
        return dataframe.loc[dataframe['entity_type'] == 'Person'].copy()
    else:
        error = "Invalid 'entity_type' value: {}".format(entity_type)
        logger.error(error)
        raise Exception(error)

def fetch_motion_features(environment, start, end, entity_type='all', include_meta_fields=False, fillna=None):
    df_cuwb_features = fetch_cuwb_data(environment,
                                       start,
                                       end,
                                       entity_type=entity_type,
                                       environment_assignment_info=True,
                                       entity_assignment_info=True)

    return extract_motion_features_from_raw(df_cuwb_features, entity_type=entity_type,
                                            include_meta_fields=include_meta_fields, fillna=fillna)


def extract_motion_features_from_raw(df_cuwb_features, entity_type='all', include_meta_fields=False, fillna=None):
    df_motion_features = extract_motion_features(
        df_position=extract_by_data_type_and_format(df_cuwb_features, data_type='position'),
        df_acceleration=extract_by_data_type_and_format(df_cuwb_features, data_type='accelerometer'),
        entity_type=entity_type,
        fillna=fillna
    )

    # TODO: Resolve assumption that each device is only assigned to a single material
    # That assumption means if a tray is reassigned to a new material, the join will
    # create duplicates records
    if include_meta_fields and len(df_cuwb_features) > 0:
        df_meta_fields = df_cuwb_features[[
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

        return df_motion_features.join(df_meta_fields, on='device_id', how='left')
    else:
        return df_motion_features


def extract_motion_features(df_position, df_acceleration, entity_type='all', fillna=None):
    f = FeatureExtraction()
    return f.extract_motion_features_for_multiple_devices(df_position, df_acceleration, entity_type, fillna=fillna)


def generate_tray_carry_groundtruth(groundtruth_csv):
    return generate_groundtruth(groundtruth_csv, groundtruth_type=ground_truth.GROUNDTRUTH_TYPE_TRAY_CARRY)


def generate_human_activity_groundtruth(groundtruth_csv):
    return generate_groundtruth(groundtruth_csv, groundtruth_type=ground_truth.GROUNDTRUTH_TYPE_HUMAN_ACTIVITY)


def generate_groundtruth(groundtruth_csv, groundtruth_type):
    try:
        df_groundtruth = load_csv(groundtruth_csv)

        if groundtruth_type == ground_truth.GROUNDTRUTH_TYPE_TRAY_CARRY:
            entity_type = 'tray'
        elif groundtruth_type == ground_truth.GROUNDTRUTH_TYPE_HUMAN_ACTIVITY:
            entity_type = 'person'
        else:
            logger.error("Unable to build groundtruth, unknown type requested: {}".format(groundtruth_type))
            return None

        valid, msg = ground_truth.validate_ground_truth(df_groundtruth, groundtruth_type=groundtruth_type)
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
                                                        entity_type=entity_type)

        if df_features is None:
            df_features = df_environment_features.copy()
        else:
            df_features = df_features.append(df_environment_features)

    df_groundtruth_features = None
    try:
        if groundtruth_type == ground_truth.GROUNDTRUTH_TYPE_TRAY_CARRY:
            df_groundtruth_features = ground_truth.combine_features_with_tray_carry_ground_truth_data(
                df_features, df_groundtruth)
        elif groundtruth_type == ground_truth.GROUNDTRUTH_TYPE_HUMAN_ACTIVITY:
            df_groundtruth_features = ground_truth.combine_features_with_human_activity_ground_truth_data(
                df_features, df_groundtruth)
    except Exception as err:
        logger.error(err)
        return None

    if df_groundtruth_features is None:
        return None

    logger.info("Tray Carry groundtruth features breakdown by device\n{}".format(
        df_groundtruth_features.fillna('NA').groupby(['device_id', 'ground_truth_state']).size()))

    return df_groundtruth_features


def generate_human_activity_model(df_groundtruth_features):
    ha = HumanActivityClassifier()
    df_groundtruth_features = df_groundtruth_features.interpolate().fillna(method='bfill')
    return ha.fit(df_groundtruth=df_groundtruth_features, scale_features=False)


def infer_human_activity(model, scaler, df_person_features):
    """
    Classifies each moment of features dataframe into a human activity state

    :param model: Human Activity carry classifier (RandomForest Model)
    :param scaler: Human Activity scaling model used to standardize features
    :param df_person_features: Dataframe with uwb data containing uwb_motion_classifiers.DEFAULT_FEATURE_FIELD_NAMES
    :return: Dataframe with uwb data containing a "predicted_tray_carry_label" column
    """
    tc = HumanActivityClassifier(model=model, feature_scaler=scaler)
    return tc.predict(df_person_features)


def generate_tray_carry_model(df_groundtruth_features, tune=False):
    tc = TrayCarryClassifier()
    df_groundtruth_features = df_groundtruth_features.interpolate().fillna(method='bfill')
    if tune:
        tc.tune(df_groundtruth=df_groundtruth_features)
        return None
    else:
        return tc.fit(df_groundtruth=df_groundtruth_features, scale_features=False)


def estimate_tray_centroids(model, scaler, df_tray_features):
    """
    Estimate the shelf location of each Tray (in x,y,z coords)

    :param model: Tray carry classifier (RandomForest Model)
    :param scaler: Tray carry scaling model used to standardize features
    :param df_tray_features: Dataframe with uwb data containing uwb_motion_classifiers.DEFAULT_FEATURE_FIELD_NAMES
    :return: Dataframe with tray centroid predictions
    """
    df_tray_features_no_movement = classifier_filter_no_movement_from_tray_features(
        model=model, scaler=scaler, df_tray_features=df_tray_features)
    #df_tray_features_no_movement = heuristic_filter_no_movement_from_tray_features(df_tray_features)
    return predict_tray_centroids(df_tray_features_no_movement=df_tray_features_no_movement)


def infer_tray_carry(model, scaler, df_tray_features):
    """
    Classifies each moment of features dataframe into a carried or not carried state

    :param model: Tray carry classifier (RandomForest Model)
    :param scaler: Tray carry scaling model used to standardize features
    :param df_tray_features: Dataframe with uwb data containing uwb_motion_classifiers.DEFAULT_FEATURE_FIELD_NAMES
    :return: Dataframe with uwb data containing a "predicted_tray_carry_label" column
    """
    tc = TrayCarryClassifier(model=model, feature_scaler=scaler)
    return tc.predict(df_tray_features)


def extract_tray_carry_events_from_inferred(df_inferred):
    """
    Extract carry events from inferred carried or not carried states (see infer_tray_carry)

    :param df_inferred: Dataframe with uwb data containing a "predicted_tray_carry_label" column
    :return: Dataframe containing carry events (device_id (tray ID), start, end)
    """
    return extract_carry_events_by_device(df_inferred)


def extract_tray_interactions(df_motion_features, df_carry_events, df_tray_centroids):
    """
    Extract carry interactions (person, tray, carry_event - FROM_SHELF/TO_SHELF/etc) from carried events (see extract_tray_carry_events_from_inferred)

    :param df_motion_features
    :param df_carry_events: Dataframe with carry events (device_id, start, end)
    :param df_tray_centroids
    :return: Dataframe containing carry interactions (person_id, device_id, start, end, carry_event)
    """
    return extract_tray_device_interactions(df_motion_features, df_carry_events, df_tray_centroids)
