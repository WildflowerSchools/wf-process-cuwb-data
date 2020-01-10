from database_connection_honeycomb import DatabaseConnectionHoneycomb
from minimal_honeycomb import MinimalHoneycombClient
import pandas as pd

def fetch_cuwb_data(
    environment_name,
    start_time,
    end_time,
    read_chunk_size=2,
    object_type_honeycomb='DEVICE',
    object_id_field_name_honeycomb='device_type',
    object_ids=['UWBTAG']
):
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
    df = df.loc[df['type'] == 'position'].copy()
    df = df.reindex(columns=[
        'serial_number',
        'timestamp_secondary',
        'x',
        'y',
        'z',
        'anchor_count',
        'quality'
    ])
    df.rename(
        columns = {
            'timestamp_secondary': 'timestamp',
            'serial_number': 'device_serial_number'
        },
        inplace=True
    )
    df['x_meters'] = df['x']/1000.0
    df['y_meters'] = df['y']/1000.0
    df['z_meters'] = df['z']/1000.0
    df.drop(
        columns=['x', 'y', 'z'],
        inplace=True
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df.dropna(subset=['timestamp'], inplace=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    return df

def fetch_cuwb_tag_metadata(
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
    return df

def fetch_entity_assignments(
    device_ids
):
    client = MinimalHoneycombClient()
    result = client.request(
        request_type="query",
        request_name='searchEntityAssignments',
        arguments={
            'query': {
                'type': 'QueryExpression!',
                'value': {
                    'field': 'device',
                    'operator': 'IN',
                    'values': device_ids
                }
            }
        },
        return_object = [
            {'data': [
                'entity_assignment_id',
                'entity_type',
                {'entity': [
                    {'... on Person': [
                        'person_id',
                        'name',
                        'first_name',
                        'last_name',
                        'nickname',
                        'short_name',
                        'person_type'
                    ]},
                    {'... on Tray': [
                        'tray_id',
                        'name',
                        'part_number',
                        'serial_number'
                    ]}
                ]},
                {'device': [
                    'device_id'
                ]},
                'start',
                'end'
            ]}
        ]
    )
    df = pd.DataFrame(result.get('data'))
    df['entity_id'] = df['entity'].apply(lambda x: x.get('tray_id', x.get('person_id')))
    df['entity_name'] = df['entity'].apply(lambda x: x.get('name'))
    df['entity_first_name'] = df['entity'].apply(lambda x: x.get('first_name'))
    df['entity_last_name'] = df['entity'].apply(lambda x: x.get('last_name'))
    df['entity_nickname'] = df['entity'].apply(lambda x: x.get('nickname'))
    df['entity_short_name'] = df['entity'].apply(lambda x: x.get('short_name'))
    df['entity_person_type'] = df['entity'].apply(lambda x: x.get('person_type'))
    df['entity_part_number'] = df['entity'].apply(lambda x: x.get('part_number'))
    df['entity_serial_number'] = df['entity'].apply(lambda x: x.get('serial_number'))
    df['device_id'] = df['device'].apply(lambda x: x.get('device_id'))
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    df = df.reindex(columns=[
        'entity_assignment_id',
        'device_id',
        'entity_id',
        'start',
        'end',
        'entity_type',
        'entity_name',
        'entity_first_name',
        'entity_last_name',
        'entity_nickname',
        'entity_short_name',
        'entity_person_type',
        'entity_part_number',
        'entity_serial_number'
    ])
    return df

def lookup_assignment(
    assignments_df,
    lookup_value,
    timestamp,
    lookup_value_field_name,
    assignment_id_field_name
):
    assignment_ids=[]
    for index, row in assignments_df.iterrows():
        if row[lookup_value_field_name] != lookup_value:
            continue
        if row['start'] > timestamp:
            continue
        if row['end'] < timestamp:
            continue
        assignment_ids.append(row[assignment_id_field_name])
    if len(assignment_ids) > 1:
        raise ValueError('More than one assignment found for obect {} at timestamp {}'.format(
            lookup_value,
            timestamp
        ))
    if len(assignment_ids) == 0:
        return None
    return assignment_ids[0]
