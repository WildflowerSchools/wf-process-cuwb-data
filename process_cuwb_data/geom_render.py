import geom_render
from minimal_honeycomb import MinimalHoneycombClient
import pandas as pd
import numpy as np
import datetime
import os
import logging

logger = logging.getLogger(__name__)

## Resample combined 3D geom collection
# frames_per_second = 10.0
# time_between_frames = datetime.timedelta(microseconds = int(round(10**6/frames_per_second)))
# num_frames = int(round((end_time - start_time)/time_between_frames))
# combined_geom_collection_3d_resampled = combined_geom_collection_3d.resample(
#     new_start_time=start_time,
#     new_frames_per_second=frames_per_second,
#     new_num_frames=num_frames,
#     progress_bar=True
# )

def create_geom_collection_3d(
    df,
    colors = {
        'Person': '#ff0000',
        'Tray': '#00ff00'
    },
    progress_bar=False,
    notebook=False
):
    # Create dictionary of 3D geom collections, one for each object in data
    logger.info('Creating dictionary of 3D geom collections, for each sensor in data: {}'.format(
        df['device_serial_number'].unique().tolist()
    ))
    geom_collection_3d_dict = dict()
    for (device_id, device_serial_number, entity_type, person_id, person_short_name, material_id, material_name), group_df in df.fillna('NA').groupby([
        'device_id',
        'device_serial_number',
        'entity_type',
        'person_id',
        'person_short_name',
        'material_id',
        'material_name'
    ]):
        entity_name = material_name
        if entity_type == 'Person':
            entity_name = person_short_name
        entity_id = material_id
        if entity_type == 'Person':
            entity_id = person_id
        logger.info('Creating 3D geom collection for {} ({}) [{} to {}]'.format(
            entity_name,
            device_serial_number,
            group_df.index.min().isoformat(),
            group_df.index.max().isoformat()
        ))
        time_index = group_df.index.to_pydatetime()
        position_values = group_df.loc[:, ['x_meters', 'y_meters', 'z_meters']].values
        coordinates = np.expand_dims(position_values, axis=1)
        geom_list = [
            geom_render.Point3D(
                coordinate_indices=[0],
                color=colors[entity_type],
                object_type=entity_type,
                object_id=entity_id,
                object_name=entity_name
            ),
            geom_render.Text3D(
                text=entity_name,
                coordinate_indices=[0],
                color=colors[entity_type],
                object_type=entity_type,
                object_id=entity_id,
                object_name=entity_name
            )
        ]
        geom_collection_3d_dict[device_id] = geom_render.GeomCollection3D(
            time_index=time_index,
            coordinates=coordinates,
            geom_list=geom_list
        )
    # Create combined 3D geom collection
    logger.info('Combining 3D geom collections into single 3D geom collection')
    combined_geom_collection_3d = geom_render.GeomCollection3D.from_geom_list(
        list(geom_collection_3d_dict.values()),
        progress_bar=progress_bar,
        notebook=notebook
    )
    return combined_geom_collection_3d


def project_onto_camera_views(
    geom_collection_3d,
    camera_info_dict
):
    ## Create 2D geom collections
    logger.info('Creating 2D geom collections from 3D geom collection, one for each camera: {}'.format(
        [camera_info['device_name'] for camera_info in camera_info_dict.values()]
    ))
    geom_collection_2d_dict = dict()
    for device_id, camera_info in camera_info_dict.items():
        logger.info('Creating 2D geom collection for camera {}'.format(
            camera_info['device_name']
        ))
        geom_collection_2d_dict[device_id] = dict()
        geom_collection_2d_dict[device_id]['device_name'] = camera_info['device_name']
        geom_collection_2d_dict[device_id]['geom'] = geom_collection_3d.project(
            rotation_vector=camera_info['rotation_vector'],
            translation_vector=camera_info['translation_vector'],
            camera_matrix=camera_info['camera_matrix'],
            distortion_coefficients=camera_info['distortion_coefficients'],
            frame_width=camera_info['image_width'],
            frame_height=camera_info['image_height']
        )
    return geom_collection_2d_dict

def write_json(
    geom_collection_2d_dict,
    output_directory='.',
    prefix='geom_2d',
    indent=None
):
    logger.info('Writing geom data to local JSON file for, one file for each camera: {}'.format(
        [geom_info['device_name'] for geom_info in geom_collection_2d_dict.values()]
    ))
    for device_id, geom_info in geom_collection_2d_dict.items():
        logger.info('Writing geom data to local JSON file for {}'.format(
            geom_info['device_name']
        ))
        path = os.path.join(
            output_directory,
            '_'.join([prefix, geom_info['device_name']]) + '.json'
        )
        with open(path, 'w') as fp:
            fp.write(geom_info['geom'].to_json(indent=indent))

def fetch_camera_info(
    device_ids,
    timestamp
):
    logger.info('Fetching camera info at timestamp {} for the following device_ids: {}'.format(
        timestamp.isoformat(),
        device_ids
    ))
    client=MinimalHoneycombClient()
    result = client.request(
        request_type='query',
        request_name='searchDevices',
        arguments={
            'query': {
                'type': 'QueryExpression!',
                'value': {
                    'field': 'device_id',
                    'operator': 'IN',
                    'values': device_ids
                }
            }
        },
        return_object = [
            {'data': [
                'device_id',
                'name',
                {'intrinsic_calibrations':[
                    'start',
                    'end',
                    'camera_matrix',
                    'distortion_coefficients',
                    'image_width',
                    'image_height'
                ]},
                {'extrinsic_calibrations':[
                    'start',
                    'end',
                    'translation_vector',
                    'rotation_vector'
                ]}
            ]}
        ]
    )
    devices = result.get('data')
    camera_info_dict = dict()
    for device in devices:
        intrinsic_calibration = extract_assignment(
            device['intrinsic_calibrations'],
            timestamp
        )
        extrinsic_calibration = extract_assignment(
            device['extrinsic_calibrations'],
            timestamp
        )
        camera_info_dict[device['device_id']] = {
            'device_name': device['name'],
            'camera_matrix': intrinsic_calibration['camera_matrix'],
            'distortion_coefficients': intrinsic_calibration['distortion_coefficients'],
            'image_width': intrinsic_calibration['image_width'],
            'image_height': intrinsic_calibration['image_height'],
            'translation_vector': extrinsic_calibration['translation_vector'],
            'rotation_vector': extrinsic_calibration['rotation_vector'],
        }
    return camera_info_dict

def extract_assignment(
    assignments,
    timestamp
):
    matched_assignments = list()
    for assignment in assignments:
        if assignment.get('start') is not None and (pd.to_datetime(assignment.get('start')).to_pydatetime() > timestamp):
            break
        if assignment.get('end') is not None and (pd.to_datetime(assignment.get('end')).to_pydatetime() < timestamp):
            break
        matched_assignments.append(assignment)
    if len(matched_assignments) == 0:
        raise ValueError('No assignments matched timestamp {}'.format(timestamp.isoformat()))
    if len(matched_assignments) > 1:
        raise ValueError('Multiple assignments matched timestamp {}'.format(timestamp.isoformat()))
    return matched_assignments[0]
