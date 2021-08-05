import multiprocessing
import numpy as np
import pandas as pd
import platform

from honeycomb_io import fetch_environment_by_name, fetch_environment_id, fetch_material_tray_devices_assignments, fetch_raw_cuwb_data
import process_pose_data

from .honeycomb_imu_data import fetch_imu_data
from .honeycomb_pose_data import pose_data_with_body_centroid as _pose_data_with_body_centroid
from .utils.io import load_csv
from .utils.log import logger
from .utils.util import filter_data_type_and_format
from .uwb_extract_data import extract_by_data_type_and_format, extract_by_entity_type
from .uwb_motion_classifier_human_activity import HumanActivityClassifier
from .uwb_motion_classifier_tray_carry import TrayCarryClassifier
from .uwb_motion_events import extract_carry_events_by_device
from .uwb_motion_features import FeatureExtraction
import process_cuwb_data.uwb_motion_ground_truth as ground_truth
from .uwb_motion_tray_interactions import infer_tray_device_interactions
from .uwb_predict_tray_centroids import classifier_filter_no_movement_from_tray_features, predict_tray_centroids


def fetch_tray_device_assignments(
        environment,
        start,
        end):
    environment = fetch_environment_by_name(environment_name=environment)
    if environment is None:
        return None

    environment_id = environment['environment_id']
    df_tray_device_assignments = fetch_material_tray_devices_assignments(environment_id, start, end)
    return df_tray_device_assignments


def fetch_cuwb_data(
    environment,
    start,
    end,
    entity_type='all',
    data_type='all'
):
    if entity_type not in ['tray', 'person', 'all']:
        raise Exception("Invalid 'entity_type' value: {}".format(entity_type))

    if data_type not in ['position', 'accelerometer', 'gyroscope', 'magnetometer', 'all']:
        raise Exception("Invalid 'data_type' value: {}".format(data_type))

    if data_type == 'all':
        imu_types_to_fetch = ['position', 'accelerometer', 'gyroscope', 'magnetometer']
    else:
        imu_types_to_fetch = [data_type]

    # When running with Jupyter, other instances of multiprocessing work w/o issue
    # But for whatever reason, this is needed or fetching cuwb data hangs
    if platform.system() == "Darwin":
        ctx = multiprocessing.get_context("spawn")
    else:
        ctx = multiprocessing

    p = ctx.Pool()

    results = []
    for imu_type in imu_types_to_fetch:
        results.append(
            p.apply_async(
                fetch_imu_data,
                kwds=dict(
                    imu_type=imu_type,
                    environment=environment,
                    start=start,
                    end=end,
                    entity_type=entity_type)))

    list_imu_data = []
    for res in results:
        if res is not None:
            list_imu_data.append(res.get())

    df_imu_data = pd.concat(list_imu_data)
    p.close()

    return extract_by_entity_type(df_imu_data, entity_type)


def fetch_cuwb_data_from_datapoints(
        environment_name,
        start_time,
        end_time,
        entity_type='all',
        data_type='raw',
        device_type='UWBTAG',
        environment_assignment_info=True,
        entity_assignment_info=True
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


def fetch_motion_features(
    environment,
    start,
    end,
    df_uwb_data=None,
    entity_type='all',
    include_meta_fields=True,
    fillna=None
):
    if df_uwb_data is None:
        df_uwb_data = fetch_cuwb_data(
            environment,
            start,
            end,
            data_type='all',
            entity_type='all'
        )

    df_position = filter_data_type_and_format(df_uwb_data, 'position')
    df_accelerometer = filter_data_type_and_format(df_uwb_data, 'accelerometer')
    df_gyroscope = filter_data_type_and_format(df_uwb_data, 'gyroscope')
    df_magnetometer = filter_data_type_and_format(df_uwb_data, 'magnetometer')
    df_motion_features = extract_motion_features(
        df_position=df_position,
        df_acceleration=df_accelerometer,
        df_gyroscope=df_gyroscope,
        df_magnetometer=df_magnetometer,
        entity_type=entity_type,
        fillna=fillna
    )

    # Add metadata fields if requested
    if include_meta_fields and\
            (len(df_position) > 0 or
             len(df_accelerometer) > 0 or
             len(df_gyroscope) > 0 or
             len(df_magnetometer) > 0):
        df_all_datatypes = pd.concat((df_position, df_accelerometer, df_gyroscope, df_magnetometer))
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
                'person_anonymized_name',
                'person_anonymized_short_name',
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


def fetch_motion_features_from_datapoints(environment,
                                          start,
                                          end,
                                          df_uwb_data=None,
                                          entity_type='all',
                                          include_meta_fields=False,
                                          fillna=None):
    if df_uwb_data is None:
        df_uwb_data = fetch_cuwb_data_from_datapoints(environment,
                                                      start,
                                                      end,
                                                      entity_type=entity_type)

    return extract_motion_features_from_raw_datapoints(df_uwb_data,
                                                       entity_type=entity_type,
                                                       include_meta_fields=include_meta_fields,
                                                       fillna=fillna)


def extract_motion_features_from_raw_datapoints(df_cuwb_features,
                                                entity_type='all',
                                                include_meta_fields=False,
                                                fillna=None):
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


def extract_motion_features(df_position,
                            df_acceleration,
                            df_gyroscope=None,
                            df_magnetometer=None,
                            entity_type='all',
                            fillna=None):
    f = FeatureExtraction()
    return f.extract_motion_features_for_multiple_devices(
        df_position=df_position,
        df_acceleration=df_acceleration,
        df_gyroscope=df_gyroscope,
        df_magnetometer=df_magnetometer,
        entity_type=entity_type,
        fillna=fillna)


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

        # Ground truth may be stored in the old datapoints table + s3 buckets
        # Check 'data_source' type and fetch accordingly
        from .uwb_motion_enum_groundtruth_data_source import GroundtruthDataSource
        if 'data_source' in group_df.columns and \
                GroundtruthDataSource(group_df.iloc[0]['data_source']) == GroundtruthDataSource.DATAPOINTS:

            # When a groundtruth example is stored in the old datapoints format, add 60 minutes offsets to start and end
            df_environment_features = fetch_motion_features_from_datapoints(environment=environment,
                                                                            start=(start - pd.DateOffset(minutes=60)),
                                                                            end=(end + pd.DateOffset(minutes=60)),
                                                                            entity_type=entity_type)
        else:
            df_environment_features = fetch_motion_features(environment=environment,
                                                            start=start,
                                                            end=end,
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
    df_groundtruth_features[FeatureExtraction.ALL_FEATURE_COLUMNS] = df_groundtruth_features[FeatureExtraction.ALL_FEATURE_COLUMNS].interpolate(
    ).fillna(method='bfill')
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

    df_tray_features[FeatureExtraction.ALL_FEATURE_COLUMNS] = df_tray_features[FeatureExtraction.ALL_FEATURE_COLUMNS].interpolate(
    ).fillna(method='bfill')

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
    df_tray_features[FeatureExtraction.ALL_FEATURE_COLUMNS] = df_tray_features[FeatureExtraction.ALL_FEATURE_COLUMNS].interpolate(
    ).fillna(method='bfill')
    return tc.predict(df_tray_features)


def extract_tray_carry_events_from_inferred(df_inferred):
    """
    Extract carry events from inferred carried or not carried states (see infer_tray_carry)

    :param df_inferred: Dataframe with uwb data containing a "predicted_tray_carry_label" column
    :return: Dataframe containing carry events (device_id (tray ID), start, end)
    """
    return extract_carry_events_by_device(df_inferred)


def infer_tray_interactions(df_motion_features, df_carry_events, df_tray_centroids, df_poses_3d=None):
    """
    Infer carry interactions (person, tray, carry_event - FROM_SHELF/TO_SHELF/etc) from carried events (see extract_tray_carry_events_from_inferred)

    :param df_motion_features
    :param df_carry_events: Dataframe with carry events (device_id, start, end)
    :param df_tray_centroids
    :param df_poses_3d
    :return: Dataframe containing carry interactions (person_id, device_id, start, end, carry_event)
    """
    df_motion_features = df_motion_features.copy()

    df_motion_features = df_motion_features[[
        'device_id',
        'entity_type',
        'tray_id',
        'tray_name',
        'material_assignment_id',
        'material_id',
        'material_name',
        'person_id',
        'person_name',
        'person_short_name',
        'person_anonymized_name',
        'person_anonymized_short_name',
        'quality',
        'x_position_smoothed',
        'y_position_smoothed',
        'z_position_smoothed'
    ]].rename(columns={'x_position_smoothed': 'x_position',
                       'y_position_smoothed': 'y_position',
                       'z_position_smoothed': 'z_position'
                       })

    df_motion_features['track_id'] = df_motion_features['device_id']
    df_motion_features['track_type'] = 'uwb_sensor'

    if df_poses_3d is not None:
        df_poses_3d_standardized = pd.DataFrame(columns=df_motion_features.columns)
        df_poses_3d_standardized['device_id'] = df_poses_3d['device_id']
        df_poses_3d_standardized['track_id'] = df_poses_3d['pose_track_3d_id']
        df_poses_3d_standardized['track_type'] = 'pose_track'
        df_poses_3d_standardized['entity_type'] = 'Person'
        df_poses_3d_standardized['person_id'] = df_poses_3d['person_id']
        df_poses_3d_standardized['person_name'] = df_poses_3d['name']
        df_poses_3d_standardized['person_short_name'] = df_poses_3d['short_name']
        df_poses_3d_standardized['person_anonymized_name'] = df_poses_3d['anonymized_name']
        df_poses_3d_standardized['person_anonymized_short_name'] = df_poses_3d['anonymized_short_name']
        df_poses_3d_standardized['x_position'] = df_poses_3d['x_position']
        df_poses_3d_standardized['y_position'] = df_poses_3d['y_position']
        df_poses_3d_standardized['z_position'] = df_poses_3d['y_position']

        df_motion_features = pd.concat([df_motion_features, df_poses_3d_standardized])

    for track_id in pd.unique(df_motion_features['track_id']):
        track_id_idx = df_motion_features['track_id'] == track_id
        df_motion_features.loc[track_id_idx, ['x_position', 'y_position', 'z_position']] = df_motion_features.loc[track_id_idx, [
            'x_position', 'y_position', 'z_position']].interpolate().fillna(method='bfill')

    return infer_tray_device_interactions(df_motion_features, df_carry_events, df_tray_centroids)


def pose_data_with_body_centroid(environment,
                                 start,
                                 end,
                                 pose_inference_id,
                                 pose_inference_base,
                                 pose_inference_subdirectory):
    environment_id = fetch_environment_id(environment_name=environment)

    df_poses_3d = process_pose_data.fetch_3d_poses_with_person_info(
        base_dir=pose_inference_base,
        environment_id=environment_id,
        pose_track_3d_identification_inference_id=pose_inference_id,
        start=start,
        end=end,
        pose_processing_subdirectory=pose_inference_subdirectory
    )

    return _pose_data_with_body_centroid(environment=environment,
                                         start=start,
                                         end=end,
                                         df_3d_pose_data=df_poses_3d)
