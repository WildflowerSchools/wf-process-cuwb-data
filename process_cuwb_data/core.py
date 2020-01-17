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

def add_assignment_ids(
    df,
    assignments_dict,
    lookup_field_name='device_id',
    assignment_field_name='assignment_id'
):
    df = df.copy()
    df[assignment_field_name] = None
    for lookup_value, assignments in assignments_dict.items():
        if len(assignments) > 0:
            lookup_boolean = (df[lookup_field_name] == lookup_value)
            for assignment in assignments:
                if pd.isnull(assignment['start']):
                    start_boolean = True
                else:
                    start_boolean = (df.index > assignment['start'])
                if pd.isnull(assignment['end']):
                    end_boolean = True
                else:
                    end_boolean = (df.index < assignment['end'])
                df.loc[
                    lookup_boolean & start_boolean & end_boolean,
                    assignment_field_name
                ] = assignment[assignment_field_name]
    return df

def fetch_tray_ids():
    client = MinimalHoneycombClient()
    result = client.request(
        request_type="query",
        request_name='entityAssignments',
        arguments=None,
        return_object = [
            {'data': [
                'entity_assignment_id',
                {'entity': [
                    {'... on Tray': [
                        'tray_id'
                    ]}
                ]}
            ]}
        ]
    )
    df = json_normalize(result.get('data'))
    df.rename(
        columns={
            'entity.tray_id': 'tray_id',
        },
        inplace=True
    )
    df.set_index('entity_assignment_id', inplace=True)
    return df

def fetch_material_assignments(
):
    client = MinimalHoneycombClient()
    result = client.request(
        request_type="query",
        request_name='materialAssignments',
        arguments=None,
        return_object = [
            {'data': [
                'material_assignment_id',
                {'tray': [
                    'tray_id'
                ]},
                'start',
                'end'
            ]}
        ]
    )
    if len(result.get('data')) == 0:
        raise ValueError('No material assignments found')
    assignments_dict = dict()
    for material_assignment in result.get('data'):
        tray_id = material_assignment['tray']['tray_id']
        assignment = {
            'material_assignment_id': material_assignment['material_assignment_id'],
            'start': material_assignment['start'],
            'end': material_assignment['end']
        }
        if tray_id in assignments_dict.keys():
            assignments_dict[tray_id].append(assignment)
        else:
            assignments_dict[tray_id] = [assignment]
    for tray_id in assignments_dict.keys():
        num_assignments = len(assignments_dict[tray_id])
        # Convert timestamp strings to Pandas datetime objects
        for assignment_index in range(num_assignments):
            assignments_dict[tray_id][assignment_index]['start'] = pd.to_datetime(
                assignments_dict[tray_id][assignment_index]['start'],
                utc=True
            )
            assignments_dict[tray_id][assignment_index]['end'] = pd.to_datetime(
                assignments_dict[tray_id][assignment_index]['end'],
                utc=True
            )
        # Sort assignment list by start time
        assignments_dict[tray_id] = sorted(
            assignments_dict[tray_id],
            key = lambda assignment: assignment['start']
        )
        # Check integrity of assignment list
        if num_assignments > 1:
            for assignment_index in range(1, num_assignments):
                if pd.isna(assignments_dict[tray_id][assignment_index - 1]['end']):
                    raise ValueError('Assignment {} starts at {} but previous assignment for this device {} starts at {} and has no end time'.format(
                        assignments_dict[tray_id][assignment_index][assignment_id_field_name],
                        assignments_dict[tray_id][assignment_index]['start'],
                        assignments_dict[tray_id][assignment_index - 1][assignment_id_field_name],
                        assignments_dict[tray_id][assignment_index - 1]['start']
                    ))
                if assignments_dict[tray_id][assignment_index]['start'] < assignments_dict[tray_id][assignment_index - 1]['end']:
                    raise ValueError('Assignment {} starts at {} but previous assignment for this device {} starts at {} and ends at {}'.format(
                        assignments_dict[tray_id][assignment_index][assignment_id_field_name],
                        assignments_dict[tray_id][assignment_index]['start'],
                        assignments_dict[tray_id][assignment_index - 1][assignment_id_field_name],
                        assignments_dict[tray_id][assignment_index - 1]['start'],
                        assignments_dict[tray_id][assignment_index - 1]['end']
                    ))
    return assignments_dict
