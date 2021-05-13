from honeycomb_io import fetch_cuwb_position_data, fetch_cuwb_accelerometer_data, fetch_cuwb_gyroscope_data, fetch_cuwb_magnetometer_data, add_device_assignment_info, add_device_entity_assignment_info, add_tray_material_assignment_info

from .utils.util import filter_entity_type


def fetch_imu_data(imu_type,
                   environment,
                   start,
                   end,
                   entity_type='all'):
    if imu_type == 'position':
        fetch = fetch_cuwb_position_data
    elif imu_type == 'accelerometer':
        fetch = fetch_cuwb_accelerometer_data
    elif imu_type == 'gyroscope':
        fetch = fetch_cuwb_gyroscope_data
    elif imu_type == 'magnetometer':
        fetch = fetch_cuwb_magnetometer_data
    else:
        raise ValueError("Unexpected IMU type: {}".format(imu_type))

    df = fetch(
        start=start,
        end=end,
        device_ids=None,
        environment_id=None,
        environment_name=environment,
        device_types=['UWBTAG'],
        output_format='dataframe',
        sort_arguments={"field": "timestamp"},
        chunk_size=20000
    )
    if len(df) == 0:
        return None

    # Add metadata
    df = add_device_assignment_info(df)
    df = add_device_entity_assignment_info(df)
    df = add_tray_material_assignment_info(df)
    # Filter on entity type
    df = filter_entity_type(df, entity_type=entity_type)

    df['type'] = imu_type
    df.reset_index(drop=True, inplace=True)
    df.set_index('timestamp', inplace=True)
    return df
