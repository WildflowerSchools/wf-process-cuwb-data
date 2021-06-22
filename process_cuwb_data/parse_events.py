import pandas as pd

from process_cuwb_data.utils.log import logger

def parse_tray_events(tray_events, time_zone='US/Central'):
    tray_events = tray_events.copy()
    tray_events['date'] = tray_events['start'].dt.tz_convert(time_zone).apply(lambda x: x.date())
    material_events_list = list()
    for (date, tray_device_id), tray_events_date_tray in tray_events.groupby(['date', 'tray_device_id']):
        material_events_list.extend(parse_tray_events_date_tray(tray_events_date_tray))
    material_events = pd.DataFrame(material_events_list)
    material_events['timestamp'] = material_events.apply(
        lambda row: row['start'] if pd.notnull(row['start']) else row['end'],
        axis=1
    )
    material_events['duration_seconds'] = (material_events['end'] - material_events['start']).dt.total_seconds()
    material_events['person_device_id'] = material_events.apply(
        lambda event: (
            event['person_device_id_from_shelf']
            if event['person_device_id_from_shelf'] == event['person_device_id_to_shelf']
            else None
        ),
        axis=1
    )
    material_events['person_name'] = material_events.apply(
        lambda event: (
            event['person_name_from_shelf']
            if event['person_name_from_shelf'] == event['person_name_to_shelf']
            else None
        ),
        axis=1
    )
    material_events['description'] = material_events.apply(
        lambda material_event: describe_material_event(
            event=material_event,
            time_zone = time_zone
        ),
        axis=1
    )
    material_events = material_events.reindex(columns=[
        'date',
        'timestamp',
        'tray_device_id',
        'material_name',
        'duration_seconds',
        'person_device_id',
        'person_name',
        'start',
        'person_device_id_from_shelf',
        'person_name_from_shelf',
        'end',
        'person_device_id_to_shelf',
        'person_name_to_shelf',
        'description'
    ])
    material_events.sort_values('timestamp', inplace=True)
    return material_events

def parse_tray_events_date_tray(tray_events_date_tray):
    tray_events_date_tray_filtered = (
        tray_events_date_tray
        .loc[tray_events_date_tray['interaction_type'].isin(['CARRYING_FROM_SHELF', 'CARRYING_TO_SHELF'])]
        .sort_values('start')
    )
    in_use = False
    material_events_list = list()
    for index, event in tray_events_date_tray_filtered.iterrows():
        interaction_type = event['interaction_type']
        if interaction_type == 'CARRYING_FROM_SHELF':
            material_events_list.append({
                'date': event['date'],
                'tray_device_id': event['tray_device_id'],
                'material_name': event['material_name'],
                'start': event['start'],
                'person_device_id_from_shelf': event['person_device_id'],
                'person_name_from_shelf': event['person_name'],
                'end': None,
                'person_device_id_to_shelf': None,
                'person_name_to_shelf': None
            })
            in_use = True
        elif interaction_type == 'CARRYING_TO_SHELF' and in_use:
            material_events_list[-1]['end'] = event['end']
            material_events_list[-1]['person_device_id_to_shelf'] = event['person_device_id']
            material_events_list[-1]['person_name_to_shelf'] = event['person_name']
            in_use = False
        elif interaction_type == 'CARRYING_TO_SHELF' and not in_use:
            material_events_list.append({
                'date': event['date'],
                'tray_device_id': event['tray_device_id'],
                'material_name': event['material_name'],
                'start': None,
                'person_device_id_from_shelf': None,
                'person_name_from_shelf': None,
                'end': event['end'],
                'person_device_id_to_shelf': event['person_device_id'],
                'person_name_to_shelf': event['person_name']
            })
            in_use = False
        else:
            raise ValueError('Encountered unexpected state: interaction type is \'{}\' and in_use is {}'.format(
                interaction_type,
                in_use
            ))
    return material_events_list


def describe_material_event(event, time_zone):
    time_string = event['timestamp'].tz_convert(time_zone).strftime('%I:%M %p')
    material_name = event['material_name']
    from_shelf_person_string = event['person_name_from_shelf'] if pd.notnull(event['person_name_from_shelf']) else 'an unknown person'
    to_shelf_person_string = event['person_name_to_shelf'] if pd.notnull(event['person_name_to_shelf']) else 'an unknown person'
    if pd.notnull(event['start']) and pd.notnull(event['end']):
        if event['duration_seconds'] > 90:
            duration_string = '{} minutes'.format(round(event['duration_seconds']/60))
        elif event['duration_seconds'] > 30:
            duration_string = '1 minute'
        else:
            duration_string = '{} seconds'.format(round(event['duration_seconds']))
        if event['person_name_from_shelf'] == event['person_name_to_shelf']:
            description_text = '{} took {} from shelf and put it back {} later'.format(
                from_shelf_person_string,
                material_name,
                duration_string
            )
        else:
            description_text = '{} took {} from shelf and {} put it back {} later'.format(
                from_shelf_person_string,
                material_name,
                to_shelf_person_string,
                duration_string
            )
    elif pd.notnull(event['start']):
        description_text = '{} took {} from shelf but never put it back'.format(
            from_shelf_person_string,
            material_name
        )
    elif pd.notnull(event['end']):
        description_text = '{} put {} back on shelf but it wasn\'t taken out previously'.format(
            to_shelf_person_string,
            material_name
        )
    else:
        raise ValueError('Unexpected state: both start and end of material event are null')
    description_text_list = list(description_text)
    description_text_list[0] = description_text_list[0].upper()
    description_text = ''.join(description_text_list)
    description = '{}: {}'.format(
        time_string,
        description_text
    )
    return description
