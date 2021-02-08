import datetime
import pandas as pd

from cachier import cachier
from database_connection_honeycomb import DatabaseConnectionHoneycomb
from minimal_honeycomb import MinimalHoneycombClient, to_honeycomb_datetime, from_honeycomb_datetime

from process_cuwb_data.utils.log import logger


#@cachier(stale_after=datetime.timedelta(days=1))
def fetch_raw_cuwb_data(
        environment_name,
        start_time,
        end_time,
        read_chunk_size=2,
        device_type='UWBTAG',
        environment_assignment_info=False,
        entity_assignment_info=False
):
    logger.info("Fetching raw CUWB tag device data for {} from {} to {}".format(environment_name, start_time, end_time))
    object_type_honeycomb = 'DEVICE'
    object_id_field_name_honeycomb = 'device_type'
    object_ids = [device_type]
    dbc = DatabaseConnectionHoneycomb(
        environment_name_honeycomb=environment_name,
        time_series_database=True,
        object_database=True,
        object_type_honeycomb=object_type_honeycomb,
        object_id_field_name_honeycomb=object_id_field_name_honeycomb,
        read_chunk_size=read_chunk_size
    )
    data = dbc.fetch_data_object_time_series(
        start_time=start_time,
        end_time=end_time,
        object_ids=object_ids
    )
    df = pd.DataFrame(data)
    if len(df) == 0:
        return df
    df.drop(
        columns=[
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
        inplace=True,
        errors='ignore'
    )
    df.rename(
        columns={
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
    df = df.join(device_data.reset_index().set_index(
        'device_serial_number'), on='device_serial_number')
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
    if environment_assignment_info:
        df = add_environment_assignment_info(df)
    if entity_assignment_info:
        df = add_entity_assignment_info(df)
    return df


def fetch_cuwb_tag_device_data(
        device_type='UWBTAG'
):
    logger.info('Fetching CUWB tag device data')
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
        return_object=[
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
    logger.info('Found CUWB device data for {} devices'.format(
        len(result.get('data'))))
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


def fetch_cuwb_tag_assignments(
        device_type='UWBTAG',
        assignment_field_name='assignments',
        assignment_id_field_name='assignment_id'
):
    logger.info('Fetching CUWB tag assignment IDs for {}'.format(
        assignment_field_name))
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
        return_object=[
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
    logger.info('Found {} {}'.format(
        len(result.get('data')),
        assignment_field_name
    ))
    if len(result.get('data')) == 0:
        raise ValueError('No devices of type {} found'.format(device_type))
    assignments_dict = {
        device['device_id']: device[assignment_field_name] for device in result.get('data')}
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
            key=lambda assignment: assignment['start']
        )
        # Check integrity of assignment list
        if num_assignments > 1:
            for assignment_index in range(1, num_assignments):
                if pd.isna(assignments_dict[device_id]
                           [assignment_index - 1]['end']):
                    raise ValueError('Assignment {} starts at {} but previous assignment for this device {} starts at {} and has no end time'.format(
                        assignments_dict[device_id][assignment_index][assignment_id_field_name],
                        assignments_dict[device_id][assignment_index]['start'],
                        assignments_dict[device_id][assignment_index -
                                                    1][assignment_id_field_name],
                        assignments_dict[device_id][assignment_index - 1]['start']
                    ))
                if assignments_dict[device_id][assignment_index]['start'] < assignments_dict[device_id][assignment_index - 1]['end']:
                    raise ValueError('Assignment {} starts at {} but previous assignment for this device {} starts at {} and ends at {}'.format(
                        assignments_dict[device_id][assignment_index][assignment_id_field_name],
                        assignments_dict[device_id][assignment_index]['start'],
                        assignments_dict[device_id][assignment_index -
                                                    1][assignment_id_field_name],
                        assignments_dict[device_id][assignment_index - 1]['start'],
                        assignments_dict[device_id][assignment_index - 1]['end']
                    ))
    return assignments_dict


def fetch_tray_ids():
    logger.info('Fetching entity assignment info to extract tray IDs')
    client = MinimalHoneycombClient()
    result = client.request(
        request_type="query",
        request_name='entityAssignments',
        arguments=None,
        return_object=[
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
    df = pd.json_normalize(result.get('data'))
    df.rename(
        columns={
            'entity.tray_id': 'tray_id',
        },
        inplace=True
    )
    logger.info(
        'Found {} entity assignments for trays'.format(
            df['tray_id'].notna().sum()))
    df.set_index('entity_assignment_id', inplace=True)
    return df


def fetch_material_assignments():
    logger.info('Fetching material assignment IDs')
    client = MinimalHoneycombClient()
    result = client.request(
        request_type="query",
        request_name='materialAssignments',
        arguments=None,
        return_object=[
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
    logger.info('Found {} material assignments'.format(
        len(result.get('data'))))
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
            key=lambda assignment: assignment['start']
        )
        # Check integrity of assignment list
        if num_assignments > 1:
            for assignment_index in range(1, num_assignments):
                if pd.isna(assignments_dict[tray_id]
                           [assignment_index - 1]['end']):
                    raise ValueError('Assignment {} starts at {} but previous assignment for this device {} starts at {} and has no end time'.format(
                        assignments_dict[tray_id][assignment_index]['material_assignment_id'],
                        assignments_dict[tray_id][assignment_index]['start'],
                        assignments_dict[tray_id][assignment_index -
                                                  1]['material_assignment_id'],
                        assignments_dict[tray_id][assignment_index - 1]['start']
                    ))
                if assignments_dict[tray_id][assignment_index]['start'] < assignments_dict[tray_id][assignment_index - 1]['end']:
                    raise ValueError('Assignment {} starts at {} but previous assignment for this device {} starts at {} and ends at {}'.format(
                        assignments_dict[tray_id][assignment_index]['material_assignment_id'],
                        assignments_dict[tray_id][assignment_index]['start'],
                        assignments_dict[tray_id][assignment_index -
                                                  1]['material_assignment_id'],
                        assignments_dict[tray_id][assignment_index - 1]['start'],
                        assignments_dict[tray_id][assignment_index - 1]['end']
                    ))
    return assignments_dict


def fetch_entity_info():
    logger.info(
        'Fetching entity assignment info to extract tray and person names')
    client = MinimalHoneycombClient()
    result = client.request(
        request_type="query",
        request_name='entityAssignments',
        arguments=None,
        return_object=[
            {'data': [
                'entity_assignment_id',
                {'entity': [
                    'entity_type: __typename',
                    {'... on Tray': [
                        'tray_id',
                        'tray_name: name'
                    ]},
                    {'... on Person': [
                        'person_id',
                        'person_name: name',
                        'person_short_name: short_name'
                    ]}
                ]}
            ]}
        ]
    )
    df = pd.json_normalize(result.get('data'))
    df.rename(
        columns={
            'entity.entity_type': 'entity_type',
            'entity.tray_id': 'tray_id',
            'entity.tray_name': 'tray_name',
            'entity.person_id': 'person_id',
            'entity.person_name': 'person_name',
            'entity.person_short_name': 'person_short_name',
        },
        inplace=True
    )
    df.set_index('entity_assignment_id', inplace=True)
    logger.info('Found {} entity assignments for trays and {} entity assignments for people'.format(
        df['tray_id'].notna().sum(),
        df['person_id'].notna().sum()
    ))
    return df


def fetch_material_names(
):
    logger.info('Fetching material assignment info to extract material names')
    client = MinimalHoneycombClient()
    result = client.request(
        request_type="query",
        request_name='materialAssignments',
        arguments=None,
        return_object=[
            {'data': [
                'material_assignment_id',
                {'material': [
                    'material_id',
                    'material_name: name'
                ]}
            ]}
        ]
    )
    df = pd.json_normalize(result.get('data'))
    df.rename(
        columns={
            'material.material_id': 'material_id',
            'material.material_name': 'material_name'
        },
        inplace=True
    )
    df.set_index('material_assignment_id', inplace=True)
    logger.info('Found {} material assignments'.format(
        df['material_id'].notna().sum()
    ))
    return df


def add_environment_assignment_info(df):
    # Fetch environment assignment IDs (devices to environment)
    environment_assignments = fetch_cuwb_tag_assignments(
        assignment_field_name='assignments',
        assignment_id_field_name='assignment_id'
    )
    # Add environment assignment IDs to dataframe
    df = add_assignment_ids(
        df=df,
        assignments_dict=environment_assignments,
        lookup_field_name='device_id',
        assignment_field_name='assignment_id'
    )
    return df


def add_entity_assignment_info(df):
    # Fetch entity assignment IDs (trays and people to devices)
    entity_assignments = fetch_cuwb_tag_assignments(
        assignment_field_name='entity_assignments',
        assignment_id_field_name='entity_assignment_id'
    )
    # Add entity assignments IDs to dataframe
    df = add_assignment_ids(
        df=df,
        assignments_dict=entity_assignments,
        lookup_field_name='device_id',
        assignment_field_name='entity_assignment_id'
    )
    # Fetch entity info (tray and person info)
    entity_info = fetch_entity_info()
    # Add entity info to dataframe
    df = df.join(entity_info, on='entity_assignment_id')
    # Fetch material assignment IDs (trays to materials)
    material_assignments = fetch_material_assignments()
    # Add material assignment IDs to dataframe
    df = add_assignment_ids(
        df=df,
        assignments_dict=material_assignments,
        lookup_field_name='tray_id',
        assignment_field_name='material_assignment_id'
    )
    # Fetch material names
    material_names = fetch_material_names()
    # Add material names to dataframe
    df = df.join(material_names, on='material_assignment_id')
    return df


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


def fetch_environment_by_name(environment_name):
    logger.info('Fetching Environments data')
    client = MinimalHoneycombClient()
    result = client.request(
        request_type="query",
        request_name="environments",
        arguments=None,
        return_object=[
            {'data':
                [
                    'environment_id',
                    'name'
                ]
             }
        ]
    )
    logger.info('Found environments data: {} records'.format(
        len(result.get('data'))))
    df = pd.DataFrame(result.get('data'))
    df = df[df['name'].str.lower().isin([environment_name.lower()])].reset_index(drop=True)
    if len(df) > 0:
        return df.loc[0]
    return None


def fetch_material_tray_devices_assignments(environment_id, start_time, end_time):

    hc_start_time = to_honeycomb_datetime(start_time)
    hc_end_time = to_honeycomb_datetime(end_time)

    logger.info('Fetching CUWB tag device data')
    client = MinimalHoneycombClient()
    result = client.request(
        request_type="query",
        request_name='searchAssignments',
        arguments={'query': {
            'type': 'QueryExpression!',
            'value':
                {
                    'operator': 'AND',
                    'children': [
                        {'operator': 'EQ', 'field': "environment", 'value': environment_id},
                        {'operator': 'LTE', 'field': "start", 'value': hc_end_time},
                        {
                            'operator': 'OR',
                            'children': [
                                {'operator': 'GTE', 'field': "end", 'value': hc_start_time},
                                {'operator': 'ISNULL', 'field': "end", 'value': hc_end_time}
                            ]
                        }
                    ]
                }
        }},
        return_object=[
            {'data': [
                {'environment': [
                    'environment_id',
                    'name'
                ]},
                'assignment_id',
                'start',
                'end',
                {'assigned': [
                    {'... on Material': [
                        'material_id',
                        'name',
                        'description',
                        {'material_assignments': [
                            'material_assignment_id',
                            'start',
                            'end',
                            {'tray': [
                                'tray_id',
                                'name',
                                {'entity_assignments': [
                                    {'device': [
                                        'device_id',
                                        'name'
                                    ]}
                                ]}
                            ]}
                        ]}
                    ]}
                ]}
            ]}
        ]
    )
    logger.info('Found material/tray/device assignments: {} records'.format(
        len(result.get('data'))))

    df_env_assignments = pd.json_normalize(result.get('data'))
    df_env_assignments = df_env_assignments[df_env_assignments['assigned.material_assignments'].notnull()]

    records = {}
    for _, env_assignment in df_env_assignments.iterrows():
        for material_assignment in env_assignment['assigned.material_assignments']:
            tray = material_assignment['tray']

            if (from_honeycomb_datetime(material_assignment['start']) > end_time or
                    (
                material_assignment['end'] is not None and
                from_honeycomb_datetime(material_assignment['end']) < start_time
            )):
                continue

            for entity_assignment in tray['entity_assignments']:
                device = entity_assignment['device']
                records[env_assignment['assignment_id']] = {
                    'start': env_assignment['start'],
                    'end': env_assignment['end'],
                    'material_id': env_assignment['assigned.material_id'],
                    'material_name': env_assignment['assigned.name'],
                    'material_description': env_assignment['assigned.description'],
                    'tray_id': tray['tray_id'],
                    'tray_name': tray['name'],
                    'tray_device_id': device['device_id'],
                    'tray_device_name': device['name'],
                }

    df = pd.DataFrame.from_dict(records, orient='index')
    return df
