import numpy as np
import pandas as pd

from honeycomb_io import fetch_environment_id, fetch_person_tag_info, fetch_persons


def pose_data_with_body_centroid(environment, start, end, df_3d_pose_data):
    # filter by person_id (note nan, perhaps we filter our known tracks w/ person_id?)
    # convert 'keypoint_coordinates_3d' to 'position_x/y/z'

    # [
    # 0: 'nose',
    # 1: 'left_eye',
    # 2: 'right_eye',
    # 3: 'left_ear',
    # 4: 'right_ear',
    # 5: 'left_shoulder',
    # 6: 'right_shoulder',
    # 7: 'left_elbow',
    # 8: 'right_elbow',
    # 9: 'left_wrist',
    # 10: 'right_wrist',
    # 11: 'left_hip',
    # 12: 'right_hip',
    # 13: 'left_knee',
    # 14: 'right_knee',
    # 15: 'left_ankle',
    # 16: 'right_ankle'
    # ]

    df_3d_pose_data = df_3d_pose_data.copy()

    keypoints = [
        {"idx": 5, "name": 'left_shoulder'},
        {"idx": 6, "name": 'right_shoulder'},
        {"idx": 11, "name": 'left_hip'},
        {"idx": 12, "name": 'right_hip'}
    ]

    environment_id = fetch_environment_id(environment_name=environment)

    cols = []
    for k in keypoints:
        cols.extend(list(map(lambda c: "{}_{}".format(k['name'], c), list("xyz"))))

    np_flattened_poses = np.array(df_3d_pose_data['keypoint_coordinates_3d'].to_list())
    np_flattened_chest_keypoints = np_flattened_poses[:, list(map(lambda x: x['idx'], keypoints)), :]
    df_flattened_chest_keypoints = pd.DataFrame(
        np_flattened_chest_keypoints.reshape(-1, 4 * 3), index=df_3d_pose_data.index, columns=cols)
    df_flattened_chest_keypoints['pose_track_3d_id'] = df_3d_pose_data['pose_track_3d_id']

    chest_keypoints_scrubbed = []
    for track in pd.unique(df_flattened_chest_keypoints['pose_track_3d_id']):
        df_track = df_flattened_chest_keypoints[df_flattened_chest_keypoints['pose_track_3d_id'] == track]
        chest_keypoints_scrubbed.append(df_track.interpolate().fillna(method='bfill'))

    df_flattened_chest_keypoints = pd.concat(chest_keypoints_scrubbed)

    df_3d_pose_data['x_position'] = df_flattened_chest_keypoints[[
        'left_shoulder_x', 'right_shoulder_x', 'left_hip_x', 'right_hip_x']].mean(axis=1)
    df_3d_pose_data['y_position'] = df_flattened_chest_keypoints[[
        'left_shoulder_y', 'right_shoulder_y', 'left_hip_y', 'right_hip_y']].mean(axis=1)
    df_3d_pose_data['z_position'] = df_flattened_chest_keypoints[[
        'left_shoulder_z', 'right_shoulder_z', 'left_hip_z', 'right_hip_z']].mean(axis=1)

    df_person_tag_info = fetch_person_tag_info(start=start, end=end, environment_id=environment_id)

    df_3d_pose_data.index = df_3d_pose_data['timestamp']
    df_3d_pose_data['device_id'] = float('nan')
    for person_id in pd.unique(df_3d_pose_data['person_id']):
        if isinstance(person_id, float) and np.isnan(person_id):
            continue

        person_details = df_person_tag_info[df_person_tag_info['person_id'] == person_id].iloc[0]
        df_3d_pose_data.loc[df_3d_pose_data['person_id'] == person_id, 'device_id'] = person_details['device_id']

    return df_3d_pose_data
