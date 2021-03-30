from process_cuwb_data.utils.log import logger


# CUWB Data Protocol: Byte size for accelerometer values
ACCELEROMETER_BYTE_SIZE = 4

# CUWB Data Protocol: Maximum integer for each byte size
CUWB_DATA_MAX_INT = {
    1: 127,
    2: 32767,
    4: 2147483647
}


def extract_by_entity_type(df, entity_type='all'):
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


def extract_by_data_type_and_format(df, data_type='raw'):
    if data_type == 'raw' or data_type is None or len(df) == 0:
        return df

    # Filter and format by entity type
    if data_type == 'position':
        return extract_position_data(df)
    elif data_type == 'accelerometer':
        return extract_accelerometer_data(df)
    elif data_type == 'gyroscope':
        return extract_accelerometer_data(df)
    elif data_type == 'magnetometer':
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
            'battery_percentage',
            'temperature',
            'x',
            'y',
            'z',
            'scale',
            'anchor_count',
            'quality',
            'smoothing'
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
            'x',
            'y',
            'z',
            'scale',
            'anchor_count',
            'quality',
            'smoothing'
        ],
        inplace=True,
        errors='ignore'
    )
    return df
