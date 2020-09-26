from .io import load_groundtruth_data
from .log import logger
from .honeycomb import fetch_raw_cuwb_data
from .tray_motion_classifiers import TrayCarryClassifier
from .tray_motion_events import extract_carry_events_by_device
from .tray_motion_features import FeatureExtraction
from .tray_motion_ground_truth import combine_features_with_ground_truth_data


# CUWB Data Protocol: Byte size for accelerometer values
ACCELEROMETER_BYTE_SIZE = 4

# CUWB Data Protocol: Maximum integer for each byte size
CUWB_DATA_MAX_INT = {
    1: 127,
    2: 32767,
    4: 2147483647
}


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

    df = filter_cuwb_data_by_entity_type(df, entity_type)
    return filter_and_format_cuwb_by_data_type(df, data_type)


def filter_cuwb_data_by_entity_type(df, entity_type='all'):
    if entity_type == 'all' or entity_type is None or len(df) == 0:
        return df

    # Filter by entity type
    if entity_type == 'tray':
        return df[df['entity_type'].eq('Tray')]
    elif entity_type == 'person':
        return df[df['entity_type'].eq('Person')]
    else:
        error = "Invalid 'entity_type' value: {}".format(entity_type)
        logger.error(error)
        raise Exception(error)


def filter_and_format_cuwb_by_data_type(df, data_type='raw'):
    if data_type == 'raw' or data_type is None or len(df) == 0:
        return df

    # Filter and format by entity type
    if data_type == 'position':
        return extract_position_data(df)
    elif data_type == 'accelerometer':
        return extract_accelerometer_data(df)
    elif data_type == 'status':
        return extract_status_data(df)
    else:
        error = "Invalid 'data_type' value: {}".format(data_type)
        logger.error(error)
        raise Exception(error)


def extract_position_data(
    df
):
    if len(df) == 0:
        return df

    df = df.loc[df['type'] == 'position'].copy()
    df['x_meters'] = df['x'] / 1000.0
    df['y_meters'] = df['y'] / 1000.0
    df['z_meters'] = df['z'] / 1000.0
    df.drop(
        columns=[
            'type',
            'battery_percentage',
            'temperature',
            'scale',
            'x',
            'y',
            'z'
        ],
        inplace=True,
        errors='ignore'
    )
    return df


def extract_accelerometer_data(
    df
):
    if len(df) == 0:
        return df

    df = df.loc[df['type'] == 'accelerometer'].copy()
    df['x_gs'] = df['x'] * df['scale'] / \
        CUWB_DATA_MAX_INT[ACCELEROMETER_BYTE_SIZE]
    df['y_gs'] = df['y'] * df['scale'] / \
        CUWB_DATA_MAX_INT[ACCELEROMETER_BYTE_SIZE]
    df['z_gs'] = df['z'] * df['scale'] / \
        CUWB_DATA_MAX_INT[ACCELEROMETER_BYTE_SIZE]
    df.drop(
        columns=[
            'type',
            'battery_percentage',
            'temperature',
            'x',
            'y',
            'z',
            'scale',
            'anchor_count',
            'quality',
            'smoothing',

        ],
        inplace=True,
        errors='ignore'
    )
    return df


def extract_status_data(
    df
):
    if len(df) == 0:
        return df

    df = df.loc[df['type'] == 'status'].copy()
    df.drop(
        columns=[
            'type',
            'x',
            'y',
            'z',
            'scale',
            'anchor_count',
            'quality',
            'smoothing',

        ],
        inplace=True,
        errors='ignore'
    )
    return df


def fetch_tray_motion_features(environment, start, end):
    df = fetch_cuwb_data(environment,
                         start,
                         end,
                         entity_type='tray',
                         data_type='raw',
                         environment_assignment_info=True,
                         entity_assignment_info=True)

    df_features = extract_tray_motion_features(
        df_position=filter_and_format_cuwb_by_data_type(df, data_type='position'),
        df_acceleration=filter_and_format_cuwb_by_data_type(df, data_type='accelerometer'),
    )

    return df_features


def extract_tray_motion_features(df_position, df_acceleration):
    f = FeatureExtraction()
    return f.extract_tray_motion_features_for_multiple_devices(df_position, df_acceleration)


def generate_tray_carry_groundtruth(environment, start, end, groundtruth_csv):
    df_groundtruth = load_groundtruth_data(groundtruth_csv)
    df_features = fetch_tray_motion_features(environment, start, end)

    df_groundtruth_features = combine_features_with_ground_truth_data(df_features, df_groundtruth)

    logger.info("Tray Carry groundtruth features breakdown by device\n{}".format(
        df_groundtruth_features.fillna('NA').groupby(['device_id', 'ground_truth_state']).size()))

    return df_groundtruth_features


def generate_tray_carry_model(groundtruth_features):
    tc = TrayCarryClassifier()
    return tc.train(df_groundtruth=groundtruth_features)


def infer_tray_carry(model, scaler, features):
    tc = TrayCarryClassifier(model=model, feature_scaler=scaler)
    return tc.inference(features)


def extract_tray_carry_events_from_inferred(inferred):
    return extract_carry_events_by_device(inferred)
