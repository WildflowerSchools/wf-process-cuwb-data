from database_connection_honeycomb import DatabaseConnectionHoneycomb
from minimal_honeycomb import MinimalHoneycombClient
import pandas as pd
from pandas.io.json import json_normalize

def fetch_cuwb_data(
    environment_name,
    start_time,
    end_time,
    read_chunk_size=2,
    device_type='UWBTAG'
):
    object_type_honeycomb = 'DEVICE'
    object_id_field_name_honeycomb ='device_type'
    object_ids=[device_type]
    dbc = DatabaseConnectionHoneycomb(
        environment_name_honeycomb = environment_name,
        time_series_database = True,
        object_database = True,
        object_type_honeycomb = object_type_honeycomb,
        object_id_field_name_honeycomb = object_id_field_name_honeycomb,
        read_chunk_size=read_chunk_size
    )
    data = dbc.fetch_data_object_time_series(
        start_time = start_time,
        end_time = end_time,
        object_ids = object_ids
    )
    df = pd.DataFrame(data)
    df.drop(
        columns = [
            'timestamp',
            'environment_name',
            'object_id',
            'memory',
            'flags',
            'minutes_remaining',
            'processor_usage',
            'network_time',
            'object_id_secondary'
        ],
        inplace=True
    )
    df.rename(
        columns = {
            'timestamp_secondary': 'timestamp',
            'serial_number': 'device_serial_number'
        },
        inplace=True
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.dropna(subset=['timestamp'], inplace=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    device_data = fetch_cuwb_tag_device_data(device_type=device_type)
    df = df.join(device_data.reset_index().set_index('device_serial_number'), on='device_serial_number')
    df = df.reindex(columns=[
        'type',
        'device_id',
        'device_part_number',
        'device_serial_number',
        'device_name',
        'device_tag_id',
        'device_mac_address',
        'battery_percentage',
        'temperature',
        'x',
        'y',
        'z',
        'scale',
        'anchor_count',
        'quality',
        'smoothing'
    ])
    return df

def fetch_cuwb_tag_device_data(
    device_type='UWBTAG'
):
    client = MinimalHoneycombClient()
    result = client.request(
        request_type="query",
        request_name='findDevices',
        arguments={
            'device_type': {
                'type': 'DeviceType',
                'value': device_type
            }
        },
        return_object = [
            {'data': [
                'device_id',
                'part_number',
                'serial_number',
                'name',
                'tag_id',
                'mac_address'
            ]}
        ]
    )
    df = pd.DataFrame(result.get('data'))
    df.rename(
        columns={
            'part_number': 'device_part_number',
            'serial_number': 'device_serial_number',
            'name': 'device_name',
            'tag_id': 'device_tag_id',
            'mac_address': 'device_mac_address'
        },
        inplace=True
    )
    df.set_index('device_id', inplace=True)
    return df

def fetch_cuwb_position_data(
    environment_name,
    start_time,
    end_time,
    read_chunk_size=2,
    device_type='UWBTAG'
):
    df = fetch_cuwb_data(
        environment_name,
        start_time,
        end_time,
        read_chunk_size,
        device_type
    )
    df = df.loc[df['type'] == 'position'].copy()
    df['x_meters'] = df['x']/1000.0
    df['y_meters'] = df['y']/1000.0
    df['z_meters'] = df['z']/1000.0
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
        inplace=True
    )
    return df

def fetch_cuwb_accelerometer_data(
    environment_name,
    start_time,
    end_time,
    read_chunk_size=2,
    device_type='UWBTAG'
):
    df = fetch_cuwb_data(
        environment_name,
        start_time,
        end_time,
        read_chunk_size,
        device_type
    )
    df = df.loc[df['type'] == 'accelerometer'].copy()
    df.drop(
        columns=[
            'type',
            'battery_percentage',
            'temperature',
            'anchor_count',
            'quality',
            'smoothing',

        ],
        inplace=True
    )
    return df

def fetch_cuwb_status_data(
    environment_name,
    start_time,
    end_time,
    read_chunk_size=2,
    device_type='UWBTAG'
):
    df = fetch_cuwb_data(
        environment_name,
        start_time,
        end_time,
        read_chunk_size,
        device_type
    )
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
        inplace=True
    )
    return df

def fetch_cuwb_tag_assignments(
    device_type='UWBTAG',
    assignment_field_name='assignments',
    assignment_id_field_name='assignment_id'
):
    client = MinimalHoneycombClient()
    result = client.request(
        request_type="query",
        request_name='findDevices',
        arguments={
            'device_type': {
                'type': 'DeviceType',
                'value': device_type
            }
        },
        return_object = [
            {'data': [
                'device_id',
                {assignment_field_name: [
                    assignment_id_field_name,
                    'start',
                    'end',
                ]}
            ]}
        ]
    )
    if len(result.get('data')) == 0:
        raise ValueError('No devices of type {} found'.format(device_type))
    assignments_dict = {device['device_id']: device[assignment_field_name] for device in result.get('data')}
    for device_id in assignments_dict.keys():
        num_assignments = len(assignments_dict[device_id])
        # Convert timestamp strings to Pandas datetime objects
        for assignment_index in range(num_assignments):
            assignments_dict[device_id][assignment_index]['start'] = pd.to_datetime(
                assignments_dict[device_id][assignment_index]['start'],
                utc=True
            )
            assignments_dict[device_id][assignment_index]['end'] = pd.to_datetime(
                assignments_dict[device_id][assignment_index]['end'],
                utc=True
            )
        # Sort assignment list by start time
        assignments_dict[device_id] = sorted(
            assignments_dict[device_id],
            key = lambda assignment: assignment['start']
        )
        # Check integrity of assignment list
        if num_assignments > 1:
            for assignment_index in range(1, num_assignments):
                if pd.isna(assignments_dict[device_id][assignment_index - 1]['end']):
                    raise ValueError('Assignment {} starts at {} but previous assignment for this device {} starts at {} and has no end time'.format(
                        assignments_dict[device_id][assignment_index][assignment_id_field_name],
                        assignments_dict[device_id][assignment_index]['start'],
                        assignments_dict[device_id][assignment_index - 1][assignment_id_field_name],
                        assignments_dict[device_id][assignment_index - 1]['start']
                    ))
                if assignments_dict[device_id][assignment_index]['start'] < assignments_dict[device_id][assignment_index - 1]['end']:
                    raise ValueError('Assignment {} starts at {} but previous assignment for this device {} starts at {} and ends at {}'.format(
                        assignments_dict[device_id][assignment_index][assignment_id_field_name],
                        assignments_dict[device_id][assignment_index]['start'],
                        assignments_dict[device_id][assignment_index - 1][assignment_id_field_name],
                        assignments_dict[device_id][assignment_index - 1]['start'],
                        assignments_dict[device_id][assignment_index - 1]['end']
                    ))
    return assignments_dict
