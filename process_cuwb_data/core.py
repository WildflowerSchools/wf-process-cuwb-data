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

def fetch_cuwb_tag_entity_assignments(
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
                {'entity_assignments': [
                    'entity_assignment_id',
                    'start',
                    'end',
                    {'entity':[
                        '__typename',
                        {'... on Person': [
                            'id:person_id',
                            'name'
                        ]},
                        {'... on Material': [
                            'id:material_id',
                            'name'
                        ]},
                        {'... on Tray': [
                            'id:tray_id',
                            'name'
                        ]},
                    ]}
                ]}
            ]}
        ]
    )
    df = json_normalize(
        result.get('data'),
        record_path='entity_assignments',
        meta='device_id'
    )
    df.rename(
        columns = {
            'entity.__typename': 'entity_type',
            'entity.id': 'entity_id',
            'entity.name': 'entity_name'
        },
        inplace=True
    )
    df['start'] = pd.to_datetime(df['start'], utc=True)
    df['end'] = pd.to_datetime(df['end'], utc=True)
    df = df.reindex(columns=[
        'entity_assignment_id',
        'start',
        'end',
        'device_id',
        'entity_type',
        'entity_id',
        'entity_name'
    ])
    df.set_index('entity_assignment_id', inplace=True)
    return df

def fetch_cuwb_tag_entity_assignments_lookup_list(
    device_type='UWBTAG'
):
    entity_assignments_df = fetch_cuwb_tag_entity_assignments(device_type)
    lookup_list = []
    for entity_assignment_id, row in entity_assignments_df.iterrows():
        lookup_list.append({
            'entity_assignment_id': entity_assignment_id,
            'device_id': row['device_id'],
            'start': row['start'],
            'end': row['end']
        })
    return lookup_list

def fetch_cuwb_tag_entity_assignments_lookup_dict(
    device_type='UWBTAG'
):
    entity_assignments_df = fetch_cuwb_tag_entity_assignments(device_type)
    lookup_dict = dict()
    for entity_assignment_id, row in entity_assignments_df.iterrows():
        device_id = row['device_id']
        assignment = {
            'entity_assignment_id': entity_assignment_id,
            'start': row['start'],
            'end': row['end']
        }
        if device_id in lookup_dict.keys():
            lookup_dict[device_id].append(assignment)
        else:
            lookup_dict[device_id] = [assignment]
    return lookup_dict

def lookup_entity_assignment_from_list(
    lookup_list,
    device_id,
    timestamp,
):
    matched_entity_assignments = list(filter(
        lambda assignment: (
            assignment['device_id'] == device_id and
            (pd.isna(assignment['start']) or assignment['start'] < timestamp) and
            (pd.isna(assignment['end']) or assignment['end'] > timestamp)
        ),
        lookup_list
    ))
    if len(matched_entity_assignments) == 0:
        return None
    if len(matched_entity_assignments) > 1:
        raise ValueError('Multiple assignments matched device ID {} and timestamp {}'.format(
            device_id,
            timestamp
        ))
    return matched_entity_assignments[0]['entity_assignment_id']

def lookup_entity_assignment_from_dict(
    lookup_dict,
    device_id,
    timestamp,
):
    if device_id not in lookup_dict.keys():
        return None
    matched_entity_assignments = list(filter(
        lambda assignment: (
            (pd.isna(assignment['start']) or assignment['start'] < timestamp) and
            (pd.isna(assignment['end']) or assignment['end'] > timestamp)
        ),
        lookup_dict[device_id]
    ))
    if len(matched_entity_assignments) == 0:
        return None
    if len(matched_entity_assignments) > 1:
        raise ValueError('Multiple assignments matched device ID {} and timestamp {}'.format(
            device_id,
            timestamp
        ))
    return matched_entity_assignments[0]['entity_assignment_id']
