import honeycomb_io
import cv_utils
import pandas as pd
import numpy as np
import datetime
import urllib.parse
import functools

from process_cuwb_data.utils.log import logger

def parse_tray_events(
    tray_events,
    environment_id=None,
    environment_name=None,
    camera_device_ids=None,
    camera_names=None,
    default_camera_device_id=None,
    default_camera_name=None,
    camera_calibrations=None,
    position_window_seconds=4,
    imputed_z_position=1.0,
    time_zone='US/Central',
    lead_in_seconds=3,
    scheme='https',
    netloc='honeycomb-ground-truth.api.wildflower-tech.org',
    endpoint='classrooms',
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    if environment_id is None and environment_name is None:
        raise ValueError('Must specify either environment ID or environment name')
    camera_info = honeycomb_io.fetch_devices(
        device_types=honeycomb_io.DEFAULT_CAMERA_DEVICE_TYPES,
        device_ids=camera_device_ids,
        names=camera_names,
        environment_id=environment_id,
        environment_name=environment_name,
        start=tray_events['start'].min(),
        end=tray_events['end'].max(),
        output_format='dataframe',
        chunk_size=chunk_size,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    camera_dict = camera_info['device_name'].to_dict()
    camera_device_ids = list(camera_dict.keys())
    if camera_calibrations is None:
        camera_calibrations = honeycomb_io.fetch_camera_calibrations(
            camera_ids=camera_device_ids,
            start=tray_events['start'].min(),
            end=tray_events['end'].max(),
            chunk_size=chunk_size,
            client=client,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    if default_camera_device_id is None:
        if default_camera_name is None:
            raise ValueError('Must specify default camera device ID or name')
        default_cameras = camera_info.loc[camera_info['device_name'] == default_camera_name]
        if len(default_cameras) == 0:
            raise ValueError('Default camera name {} not found'.format(
                default_camera_name
            ))
        if len(default_cameras) > 1:
            raise ValueError('More than one camera with default camera name {} found'.format(
                default_camera_name
            ))
        default_camera_device_id = default_cameras.index[0]
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
    best_camera_partial = functools.partial(
        best_camera,
        default_camera_device_id=default_camera_device_id,
        environment_id=environment_id,
        environment_name=environment_name,
        camera_dict=camera_dict,
        camera_calibrations=camera_calibrations,
        position_window_seconds=position_window_seconds,
        imputed_z_position=imputed_z_position,
        chunk_size=chunk_size,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    material_events['best_camera_device_id_from_shelf'] = material_events.apply(
        lambda event: best_camera_partial(
            timestamp=event['start'],
            tray_device_id=event['tray_device_id'],
        ) if pd.notnull(event['start']) else None,
        axis=1
    )
    material_events['best_camera_name_from_shelf'] = material_events['best_camera_device_id_from_shelf'].apply(
        lambda camera_device_id: camera_dict.get(camera_device_id)
    )
    material_events['best_camera_device_id_to_shelf'] = material_events.apply(
        lambda event: best_camera_partial(
            timestamp=event['end'],
            tray_device_id=event['tray_device_id'],
        ) if pd.notnull(event['end']) else None,
        axis=1
    )
    material_events['best_camera_name_to_shelf'] = material_events['best_camera_device_id_to_shelf'].apply(
        lambda camera_device_id: camera_dict.get(camera_device_id)
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
        lambda event: describe_material_event(
            timestamp=event['timestamp'],
            material_name=event['material_name'],
            start=event['start'],
            person_name_from_shelf=event['person_name_from_shelf'],
            end=event['end'],
            person_name_to_shelf=event['person_name_to_shelf'],
            duration_seconds=event['duration_seconds'],
            time_zone=time_zone
        ),
        axis=1
    )
    material_events['description_html'] = material_events.apply(
        lambda event: describe_material_event_html(
            timestamp=event['timestamp'],
            material_name=event['material_name'],
            start=event['start'],
            person_name_from_shelf=event['person_name_from_shelf'],
            end=event['end'],
            best_camera_name_from_shelf=event['best_camera_name_from_shelf'],
            person_name_to_shelf=event['person_name_to_shelf'],
            best_camera_name_to_shelf=event['best_camera_name_to_shelf'],
            duration_seconds=event['duration_seconds'],
            environment_id=environment_id,
            time_zone=time_zone,
            lead_in_seconds=lead_in_seconds,
            scheme=scheme,
            netloc=netloc,
            endpoint=endpoint
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
        'best_camera_device_id_from_shelf',
        'best_camera_name_from_shelf',
        'end',
        'person_device_id_to_shelf',
        'person_name_to_shelf',
        'best_camera_device_id_to_shelf',
        'best_camera_name_to_shelf',
        'description',
        'description_html'
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

def describe_material_event(
    timestamp,
    material_name,
    start,
    person_name_from_shelf,
    end,
    person_name_to_shelf,
    duration_seconds,
    time_zone
):
    time_string = timestamp.tz_convert(time_zone).strftime('%I:%M %p')
    from_shelf_person_string = person_name_from_shelf if pd.notnull(person_name_from_shelf) else 'an unknown person'
    to_shelf_person_string = person_name_to_shelf if pd.notnull(person_name_to_shelf) else 'an unknown person'
    if pd.notnull(start) and pd.notnull(end):
        if duration_seconds > 90:
            duration_string = '{} minutes'.format(round(duration_seconds/60))
        elif duration_seconds > 30:
            duration_string = '1 minute'
        else:
            duration_string = '{} seconds'.format(round(duration_seconds))
        if person_name_from_shelf == person_name_to_shelf:
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
    elif pd.notnull(start):
        description_text = '{} took {} from shelf but never put it back'.format(
            from_shelf_person_string,
            material_name
        )
    elif pd.notnull(end):
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

def describe_material_event_html(
    timestamp,
    material_name,
    start,
    person_name_from_shelf,
    best_camera_name_from_shelf,
    end,
    person_name_to_shelf,
    best_camera_name_to_shelf,
    duration_seconds,
    environment_id,
    time_zone,
    lead_in_seconds=3,
    scheme='https',
    netloc='honeycomb-ground-truth.api.wildflower-tech.org',
    endpoint='classrooms'
):
    time_string = timestamp.tz_convert(time_zone).strftime('%I:%M %p')
    from_shelf_person_string = person_name_from_shelf if pd.notnull(person_name_from_shelf) else 'an unknown person'
    to_shelf_person_string = person_name_to_shelf if pd.notnull(person_name_to_shelf) else 'an unknown person'
    url_from_shelf = event_url(
        environment_id=environment_id,
        timestamp=start - datetime.timedelta(seconds=lead_in_seconds),
        camera_name=best_camera_name_from_shelf,
        scheme=scheme,
        netloc=netloc,
        endpoint=endpoint
    )
    url_to_shelf = event_url(
        environment_id=environment_id,
        timestamp=end - datetime.timedelta(seconds=lead_in_seconds),
        camera_name=best_camera_name_to_shelf,
        scheme=scheme,
        netloc=netloc,
        endpoint=endpoint
    )
    if pd.notnull(start) and pd.notnull(end):
        if duration_seconds > 90:
            duration_string = '{} minutes'.format(round(duration_seconds/60))
        elif duration_seconds > 30:
            duration_string = '1 minute'
        else:
            duration_string = '{} seconds'.format(round(duration_seconds))
        if person_name_from_shelf == person_name_to_shelf:
            description_text = '<a href=\"{}\">{} took {} from shelf</a> and <a href=\"{}\">put it back {} later</a>'.format(
                url_from_shelf,
                from_shelf_person_string,
                material_name,
                url_to_shelf,
                duration_string
            )
        else:
            description_text = '<a href=\"{}\">{} took {} from shelf</a> and <a href=\"{}\">{} put it back {} later</a>'.format(
                url_from_shelf,
                from_shelf_person_string,
                material_name,
                url_to_shelf,
                to_shelf_person_string,
                duration_string
            )
    elif pd.notnull(start):
        description_text = '<a href=\"{}\">{} took {} from shelf</a> but never put it back'.format(
            url_from_shelf,
            from_shelf_person_string,
            material_name
        )
    elif pd.notnull(end):
        description_text = '<a href=\"{}\">{} put {} back on shelf</a> but it wasn\'t taken out previously'.format(
            url_to_shelf,
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


def event_url(
    environment_id,
    timestamp,
    camera_name,
    scheme='https',
    netloc='honeycomb-ground-truth.api.wildflower-tech.org',
    endpoint='classrooms'
):
    if timestamp is None or camera_name is None:
        return None
    date_string = pd.to_datetime(timestamp, utc=True).strftime('%Y-%m-%d')
    time_string = pd.to_datetime(timestamp, utc=True).strftime('%H:%M:%S')
    url = urllib.parse.urlunparse((
        scheme,
        netloc,
        '/'.join([
            endpoint,
            environment_id,
            date_string
        ]),
        None,
        urllib.parse.urlencode({
            'device': camera_name,
            'time': time_string
        }),
        None
    ))
    return url

def best_camera(
    timestamp,
    tray_device_id,
    default_camera_device_id,
    environment_id=None,
    environment_name=None,
    camera_dict=None,
    camera_calibrations=None,
    position_window_seconds=4,
    imputed_z_position = 1.0,
    chunk_size=100,
    client=None,
    uri=None,
    token_uri=None,
    audience=None,
    client_id=None,
    client_secret=None
):
    if camera_dict is None:
        if environment_id is None and environment_name is None:
            raise ValueError('If camera dictionary is not specified, must specify either environment ID or environment name')
        camera_info = honeycomb_io.fetch_devices(
            device_types=honeycomb_io.DEFAULT_CAMERA_DEVICE_TYPES,
            environment_id=environment_id,
            environment_name=environment_name,
            start=timestamp,
            end=timestamp,
            output_format='dataframe',
            chunk_size=chunk_size,
            client=client,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
        camera_dict = camera_info['device_name'].to_dict()
    camera_device_ids = list(camera_dict.keys())
    if camera_calibrations is None:
        camera_calibrations = honeycomb_io.fetch_camera_calibrations(
            camera_ids=camera_device_ids,
            start=timestamp,
            end=timestamp,
            chunk_size=chunk_size,
            client=client,
            uri=uri,
            token_uri=token_uri,
            audience=audience,
            client_id=client_id,
            client_secret=client_secret
        )
    position_window_start = timestamp - datetime.timedelta(seconds=position_window_seconds/2)
    position_window_end = timestamp + datetime.timedelta(seconds=position_window_seconds/2)
    position_data = honeycomb_io.fetch_cuwb_position_data(
        start=position_window_start,
        end=position_window_end,
        device_ids=[tray_device_id],
        environment_id=None,
        environment_name=None,
        device_types=['UWBTAG'],
        output_format='dataframe',
        sort_arguments=None,
        chunk_size=1000,
        client=client,
        uri=uri,
        token_uri=token_uri,
        audience=audience,
        client_id=client_id,
        client_secret=client_secret
    )
    position = np.nanmedian(position_data.loc[:, ['x', 'y', 'z']].values, axis=0)
    if imputed_z_position is not None:
        position[2] = imputed_z_position
    view_data_list = list()
    for camera_device_id, camera_calibration in camera_calibrations.items():
        camera_name = camera_dict.get(camera_device_id)
        camera_position = cv_utils.extract_camera_position(
            rotation_vector=camera_calibration['rotation_vector'],
            translation_vector=camera_calibration['translation_vector']
        )
        distance_from_camera = np.linalg.norm(np.subtract(
            position,
            camera_position
        ))
        image_position = cv_utils.project_points(
            object_points=position,
            rotation_vector=camera_calibration['rotation_vector'],
            translation_vector=camera_calibration['translation_vector'],
            camera_matrix=camera_calibration['camera_matrix'],
            distortion_coefficients=camera_calibration['distortion_coefficients'],
            remove_behind_camera=True,
            remove_outside_frame=True,
            image_corners=np.asarray([
                [0.0, 0.0],
                [camera_calibration['image_width'], camera_calibration['image_height']]
            ])
        )
        image_position = np.squeeze(image_position)
        if np.all(np.isfinite(image_position)):
            in_frame = True
            distance_from_image_center = np.linalg.norm(np.subtract(
                image_position,
                [camera_calibration['image_width']/2, camera_calibration['image_height']/2]
            ))
            in_middle = (
                image_position[0] > camera_calibration['image_width']*(1.0/10.0) and
                image_position[0] < camera_calibration['image_width']*(9.0/10.0) and
                image_position[1] > camera_calibration['image_height']*(1.0/10.0) and
                image_position[1] < camera_calibration['image_height']*(9.0/10.0)
            )
        else:
            in_frame=False
            distance_from_image_center=None
            in_middle=False
        view_data_list.append({
            'camera_device_id': camera_device_id,
            'camera_name': camera_name,
            'position': position,
            'distance_from_camera': distance_from_camera,
            'image_position': image_position,
            'distance_from_image_center': distance_from_image_center,
            'in_frame': in_frame,
            'in_middle': in_middle,
        })
    view_data = pd.DataFrame(view_data_list).set_index('camera_device_id')
    if not view_data['in_frame'].any():
        best_camera_device_id = default_camera_device_id
    elif not view_data['in_middle'].any():
        best_camera_device_id = view_data.sort_values('distance_from_image_center').index[0]
    else:
        best_camera_device_id = view_data.loc[view_data['in_middle']].sort_values('distance_from_camera').index[0]
    best_camera_name = camera_dict.get(best_camera_device_id)
    return best_camera_device_id