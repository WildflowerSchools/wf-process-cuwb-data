from functools import partial
import multiprocessing
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import sys
import time

from process_cuwb_data.utils.log import logger
from process_cuwb_data.utils.util import dataframe_tuple_columns_to_underscores
from process_cuwb_data.uwb_motion_enum_interaction_types import InteractionType


# TODO: Ignoring z-axis when computing distance for now, reconsider after further testing CUWB anchors
DIMENSIONS_WHEN_COMPUTING_CHILD_TRAY_DISTANCE = 2
DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE = 2
CARRY_EVENT_DISTANCE_BETWEEN_TRAY_AND_PERSON = 1.25
CARRY_EVENT_DISTANCE_BETWEEN_TRAY_AND_SHELF = 0.5


def map_column_name_to_dimension_space(column_name, num_dimensions):
    dims = ['x', 'y', 'z']
    return list(map(lambda d: "{}_{}".format(d, column_name), dims[0:num_dimensions]))


def modify_carry_events_with_track_ids(df_carry_events):
    """
    Generate a set of track IDs for each carry event.

    :param df_carry_events:
    :return: A modified df_carry_events dataframe with numeric track_ids
    """
    df_carry_events_modified = df_carry_events.copy()
    df_carry_events_with_track_id = df_carry_events_modified.reset_index(drop=True).sort_values(
        'start').assign(tray_track_id=range(len(df_carry_events_modified.index)))
    return df_carry_events_with_track_id


def augment_carry_events_start_and_end_times(df_carry_events_with_track_ids, num_seconds=1.0):
    """
    Add/subtract 1 second from tray
    carry start and end. Assumption is the tray will be more stable when observed a second
    before and after tray carrying.

    :param df_carry_events_with_track_ids:
    :param num_seconds:
    :return: A modified df_carry_events dataframe with shifted start/end times
    """
    df_carry_events_augmented = df_carry_events_with_track_ids.copy()
    df_carry_events_augmented['start_augmented'] = df_carry_events_augmented['start'] - \
        pd.Timedelta(seconds=num_seconds)
    df_carry_events_augmented['end_augmented'] = df_carry_events_augmented['end'] + pd.Timedelta(seconds=num_seconds)
    return df_carry_events_augmented


def get_estimated_tray_location_from_carry_events(df_features, df_carry_events):
    df_tray_features = df_features[df_features['entity_type'] == 'Tray']

    # Fudge start/stop times to get a better guess at resting tray locations
    df_carry_events_with_track_ids_and_augmented_times = augment_carry_events_start_and_end_times(df_carry_events)

    carry_events_with_positions = []
    position_cols = map_column_name_to_dimension_space(
        'position', DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE)
    for _, row in df_carry_events_with_track_ids_and_augmented_times.iterrows():
        # TODO: Rather than use the start_augmented time, consider trying to find
        # the moment between 'start_augmented' and actual 'start' when a given
        # tray is nearest to a given tray's centroid or shelf location
        device_id_mask = (df_tray_features['device_id'] == row['device_id'])
        if row['start_augmented'] in df_tray_features.index:
            start_mask = (df_tray_features.index == row['start_augmented']) & device_id_mask
        elif row['start'] in df_tray_features.index:
            start_mask = (df_tray_features.index == row['start']) & device_id_mask
        else:
            logger.warning(
                "Couldn't determine a carry event start time for '{}', skipping carry event".format(
                    df_tray_features['device_id']))
            continue

        if row['end_augmented'] in df_tray_features.index:
            end_mask = (df_tray_features.index == row['end_augmented']) & device_id_mask
        elif row['end'] in df_tray_features.index:
            end_mask = (df_tray_features.index == row['end']) & device_id_mask
        else:
            logger.warning(
                "Couldn't determine a carry event end time for '{}', skipping carry event".format(
                    df_tray_features['device_id']))
            continue

        cols = [*['device_id'], *position_cols]
        df_start_position = df_tray_features.loc[start_mask][cols]
        if len(df_start_position) == 0:
            logger.warning(
                "Expected a carry event for '{}' at time {} to exist but none found, skipping carry event".format(
                    df_tray_features['device_id'], row['start']))
            continue

        df_start_position = df_start_position.assign(carry_moment='start')
        df_start_position = df_start_position.assign(tray_track_id=row['tray_track_id'])
        df_start_position.index = [row['start']]

        df_end_position = df_tray_features.loc[end_mask][cols]
        if len(df_end_position) == 0:
            logger.warning(
                "Expected a carry event for '{}' at time {} to exist but none found, skipping carry event".format(
                    df_tray_features['device_id'], row['start']))
            continue

        df_end_position = df_end_position.assign(carry_moment='end')
        df_end_position = df_end_position.assign(tray_track_id=row['tray_track_id'])
        df_end_position.index = [row['end']]

        carry_events_with_positions.append(df_start_position)
        carry_events_with_positions.append(df_end_position)

    df_positions_for_carry_event_moments = pd.concat(carry_events_with_positions)
    return df_positions_for_carry_event_moments


def filter_features_by_carry_events_and_split_by_device_type(df_features, df_carry_events_with_track_ids):
    """
    Before computing distances between people and trays, filter out all candidate uwb features
    based on tray carry times. Then split between device types (Tray and Person)

    :param df_features:
    :param df_carry_events_with_track_ids:
    :return: 2 item Tuple
             first index containing a filtered people dataframe
             second index containing a filtered tray dataframe with track IDs
    """

    # First synchronize all devices with the same number of time indices
    df_unique_indexes = pd.DataFrame(index=pd.MultiIndex.from_product(
        [df_features.index.unique(), df_features['device_id'].unique()]).set_names(['timestamp', 'device_id'])).sort_index()
    df_features_with_device_id = df_features.set_index(keys='device_id', append=True)
    df_features_with_device_id.index = df_features_with_device_id.index.set_names(['timestamp', 'device_id'])
    _, df_features_with_aligned_indexes = df_unique_indexes.align(df_features_with_device_id, join='left', axis=0)
    df_features_with_aligned_indexes = df_features_with_aligned_indexes.sort_index()

    # Next split people and tray features apart
    time_slices = []  # People will be sliced by time only
    time_and_tray_slices = []  # Trays will be sliced by time and device_id. Trays are also assigned a TrackID
    for idx, row in df_carry_events_with_track_ids.iterrows():
        time_slices.append(slice(row['start'], row['end']))
        time_and_tray_slices.append([(slice(row['start'], row['end']), row['device_id']), row['tray_track_id']])

    df_people_features = df_features_with_aligned_indexes[df_features_with_aligned_indexes['entity_type'] == 'Person']
    df_tray_features = df_features_with_aligned_indexes[df_features_with_aligned_indexes['entity_type'] == 'Tray']

    df_people_features_sliced = pd.concat(list(map(lambda s: df_people_features.loc[s, :], time_slices)))
    df_tray_features_sliced_with_track_id = pd.concat(
        list(map(lambda s: df_tray_features.loc[s[0], :].assign(tray_track_id=s[1]), time_and_tray_slices)))

    return df_people_features_sliced.reset_index(level=1), df_tray_features_sliced_with_track_id.reset_index(level=1)


def people_trays_cdist_iterable(idx, _df_people, _df_trays, v_count, v_start, lock, size):
    """
    Background runnable function to compute distances between people and tray.

    :param idx: Dataframe lookup index (time based)
    :param _df_people: People dataframe
    :param _df_trays: Trays dataframe
    :param v_count: Iteration count
    :param v_start: Timer for logging progress
    :param lock: Multiprocessing lock for variable manipulation
    :param size: Total number of records for processing
    :return: Dataframe of joined people and trays with computed distance
    """
    with lock:
        if v_count.value % 1000 == 0:
            logger.info("Computing tray <-> people distances: Processed {}/{} - Time {}s".format(v_count.value,
                                                                                                 size, time.time() - v_start.value))
            sys.stdout.flush()
            v_start.value = time.time()

        v_count.value += 1

    df_people_by_idx = _df_people.loc[[idx]]
    df_trays_by_idx = _df_trays.loc[[idx]]

    position_cols = map_column_name_to_dimension_space(
        'position', DIMENSIONS_WHEN_COMPUTING_CHILD_TRAY_DISTANCE)

    df_people_and_trays = df_people_by_idx.join(df_trays_by_idx, how='inner', lsuffix='_person', rsuffix='_tray')
    distances = cdist(df_people_by_idx[position_cols].to_numpy(),
                      df_trays_by_idx[position_cols].to_numpy(),
                      metric='euclidean')

    return df_people_and_trays.assign(person_tray_distance=distances.flatten())


def generate_person_tray_distances(df_people_features, df_tray_features):
    """
    Use multi-processing to generate distances between candidate people and trays
    across all recorded features (computationally heavy because distance is generated
    for every timestamp between every person/tray device)

    :param df_people_features:
    :param df_tray_features:
    :return:
    """
    p = multiprocessing.Pool()
    m = multiprocessing.Manager()

    lock = m.Lock()
    start = time.time()
    v_count = m.Value('i', 0)  # Keep track of iterations
    v_start = m.Value('f', start)  # Share timer object
    time_indexes = df_people_features.index.unique(level=0)

    df_person_tray_distances = pd.concat(
        p.map(
            partial(
                people_trays_cdist_iterable,
                _df_people=df_people_features,
                _df_trays=df_tray_features,
                v_count=v_count,
                v_start=v_start,
                lock=lock,
                size=len(time_indexes)),
            time_indexes))
    logger.info("Finished computing tray <-> people distances: {}/{} - Total time: {}s".format(len(time_indexes),
                                                                                               len(time_indexes), time.time() - start))

    p.close()
    p.join()

    return df_person_tray_distances


def aggregate_clean_filter_person_tray_distances(df_person_tray_distances, df_carry_events_with_track_ids):
    """
    Determine the aggregated median distance between each person and tray. Note
    that this function also handles tracks pose data tracks which might have time
    gaps. To handle this there is logic to clean and filter the data structure
    before returning the nearest person<->tray candidates.

    :param df_person_tray_distances:
    :param df_carry_events_with_track_ids:
    :return: Dataframe (df_carry_events_distances_from_people)

    e.g.
    tray_track_id,device_id_person,person_name_person,device_id_tray,material_name_tray,person_tray_distance_median,person_track_length_seconds,person_tray_distance_min,person_tray_distance_max
    0,34da139d-c6bb-493f-b775-f9a568f6d20b,Bert,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory Bin,6.58958,4.00000,6.04818,6.92025
    0,5af061c7-8d6c-4c8d-8f8e-26fc8fe09ae3,Ernie,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory Bin,5.20226,4.00000,4.53189,6.20005
    0,9e2799f6-56a8-4878-8df7-401b2759408c,Grover,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory Bin,5.01834,4.00000,4.57632,5.15572
    0,bbf3f7d4-994a-4418-93fc-ec0c009cc90a,Cookie,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory Bin,0.33810,4.00000,0.02427,0.60723
    0,nan,nan,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory Bin,2.30686,4.00000,1.45119,3.68772
    0,nan,nan,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory Bin,5.05548,4.00000,4.56820,5.25454
    0,d13dad39-cba0-48f0-b3d1-29e9ec92ec56,Elmo,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory Bin,2.64016,4.00000,1.72280,4.04284
    0,bbf3f7d4-994a-4418-93fc-ec0c009cc90a,Cookie,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory Bin,0.36287,3.60000,0.09385,0.57656
    1,...
    1,...
    1,...
    1,...
    1,...
    2,...
    2,...
    2,...
    2,...
    ...
    """

    # Group and aggregate each person and tray-carry-track combination to get the mean distance during the carry event
    # FYI: Grouping creates a dataframe with multiple column axis
    df_person_tray_distances['timestamp'] = df_person_tray_distances.index
    df_person_tray_distances_aggregated = df_person_tray_distances.groupby(
        [
            'tray_track_id',
            'device_id_person',
            'person_id_person',
            'person_name_person',
            'person_short_name_person',
            'person_anonymized_name_person',
            'person_anonymized_short_name_person',
            'track_id_person',  # Used to differentiate pose_tracks from uwb_tracks
            'track_type_person',  # Used to differentiate pose_tracks from uwb_tracks
            'device_id_tray',
            'material_name_tray']).agg(
        {
            'timestamp': [
                'min',
                'max'
            ],
            'person_tray_distance': [
                'median',
                'min',
                'max']}).reset_index()
    df_person_tray_distances_aggregated.columns = df_person_tray_distances_aggregated.columns.to_flat_index()
    dataframe_tuple_columns_to_underscores(df_person_tray_distances_aggregated, inplace=True)
    df_person_tray_distances_aggregated.rename(
        columns={
            'timestamp_min': 'start',
            'timestamp_max': 'end'},
        inplace=True)
    df_person_tray_distances_aggregated['person_track_length_seconds'] = (
        df_person_tray_distances_aggregated['end'] -
        df_person_tray_distances_aggregated['start']).dt.total_seconds()

    # Example pre-filtered df_person_tray_distances_aggregated dataframe
    #
    # 1) In this example, a tray is carried for 4 seconds
    # 2) This dataset includes both uwb data as well as pose track data
    # 3) Notice that Cookie has one uwb track and two pose tracks. Pose tracks can get interruped, so we might see multiple over a given time frame. It's for this reason we capture a start and end time.
    # 4) Also notice there are two tracks for Elmo, one uwb based and one pose based
    # 5) Elmo's UWB track appears closest to the given tray/material (0.33810 meters median). It is just a hair closer than what the pose track came up with (0.36287 meters median)
    #
    # tray_track_id,device_id_person,person_name_person,track_id_person,track_type_person,device_id_tray,material_name_tray,start,end,person_tray_distance_median,person_tray_distance_min,person_tray_distance_max,person_track_length_seconds
    # 0,34da139d-c6bb-493f-b775-f9a568f6d20b,Bert,34da139d-c6bb-493f-b775-f9a568f6d20b,uwb_sensor,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory Bin,2021-04-20 14:01:26.300000+00:00,2021-04-20 14:01:30.300000+00:00,6.58958,6.04818,6.92025,4.0
    # 0,5af061c7-8d6c-4c8d-8f8e-26fc8fe09ae3,Ernie,5af061c7-8d6c-4c8d-8f8e-26fc8fe09ae3,uwb_sensor,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory Bin,2021-04-20 14:01:26.300000+00:00,2021-04-20 14:01:30.300000+00:00,5.20226,4.53189,6.20005,4.0
    # 0,9e2799f6-56a8-4878-8df7-401b2759408c,Oscar,9e2799f6-56a8-4878-8df7-401b2759408c,uwb_sensor,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory Bin,2021-04-20 14:01:26.300000+00:00,2021-04-20 14:01:30.300000+00:00,5.01834,4.57632,5.15572,4.0
    # 0,bbf3f7d4-994a-4418-93fc-ec0c009cc90a,Elmo,8d6d4bec000e4168b371bfcbae4116f6,pose_track,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory Bin,2021-04-20 14:01:26.700000+00:00,2021-04-20 14:01:30.300000+00:00,0.36287,0.09385,0.57656,3.6
    # 0,bbf3f7d4-994a-4418-93fc-ec0c009cc90a,Elmo,bbf3f7d4-994a-4418-93fc-ec0c009cc90a,uwb_sensor,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory Bin,2021-04-20 14:01:26.300000+00:00,2021-04-20 14:01:30.300000+00:00,0.33810,0.02427,0.60723,4.0
    # 0,d13dad39-cba0-48f0-b3d1-29e9ec92ec56,Cookie,9a5def9142b244c4870826e311fbbc08,pose_track,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory Bin,2021-04-20 14:01:26.300000+00:00,2021-04-20 14:01:30.300000+00:00,2.30686,1.45119,3.68772,4.0
    # 0,d13dad39-cba0-48f0-b3d1-29e9ec92ec56,Cookie,b3484c808e4c4672a79906140430742f,pose_track,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory Bin,2021-04-20 14:01:26.300000+00:00,2021-04-20 14:01:30.300000+00:00,5.05548,4.56820,5.25454,4.0
    # 0,d13dad39-cba0-48f0-b3d1-29e9ec92ec56,Cookie,d13dad39-cba0-48f0-b3d1-29e9ec92ec56,uwb_sensor,f37dd610-7335-4355-b706-0a1ea3f35519,Sensory
    # Bin,2021-04-20 14:01:26.300000+00:00,2021-04-20
    # 14:01:30.300000+00:00,2.64016,1.72280,4.04284,4.0

    df_carry_events_with_track_ids['track_length_seconds'] = (
        df_carry_events_with_track_ids['end'] -
        df_carry_events_with_track_ids['start']).dt.total_seconds()

    split_track_row_ids = []

    # Step-1 - cleanup: Find overlapping people across the 'pose track' records (i.e. Cookie) and set person to nan
    # TODO: This may be overly aggressive, we may want to only delete specific overlapping moments...
    # TODO: ...rather than deleting whole tracks if there is any moment of overlap
    for _, carry_event_row in df_carry_events_with_track_ids.iterrows():
        tray_track_id = carry_event_row['tray_track_id']
        pose_tracks_filter_idx =\
            (df_person_tray_distances_aggregated['track_type_person'] == 'pose_track') &\
            (df_person_tray_distances_aggregated['tray_track_id'] == tray_track_id)

        overlapping_people_ids = []

        df_person_tray_distances_aggregated_pose_tracks_only = df_person_tray_distances_aggregated[
            pose_tracks_filter_idx]
        for person_id in pd.unique(df_person_tray_distances_aggregated_pose_tracks_only['device_id_person']):
            if person_id == float(np.nan):
                continue

            df_person_tracks_filter_idx = df_person_tray_distances_aggregated_pose_tracks_only[
                'device_id_person'] == person_id
            df_person_tracks = df_person_tray_distances_aggregated_pose_tracks_only[df_person_tracks_filter_idx]

            for idx, row in df_person_tracks.iterrows():
                overlapping_start_idx = (
                    row['start'] >= df_person_tracks['start']) & (
                    row['start'] <= df_person_tracks['end'])
                if len(overlapping_start_idx) > 1:
                    overlapping_people_ids.extend(df_person_tracks.loc[overlapping_start_idx, 'track_id_person'].values)

                df_overlapping_end_idx = (
                    row['end'] >= df_person_tracks['start']) & (
                    row['end'] <= df_person_tracks['end'])
                if len(df_overlapping_end_idx) > 1:
                    overlapping_people_ids.extend(
                        df_person_tracks.loc[df_overlapping_end_idx, 'track_id_person'].values)

        overlapping_idx = df_person_tray_distances_aggregated['track_id_person'].isin(overlapping_people_ids)
        df_person_tray_distances_aggregated.loc[overlapping_idx, [
            'device_id_person',
            'person_id_person',
            'person_name_person',
            'person_short_name_person',
            'person_anonymized_name_person',
            'person_anonymized_short_name_person'
        ]] = float(np.nan)

    # Filter step-2 (A):
    # In preparation for handling split pose tracks, begin gathering groups of
    # associated track_ids. Step A is to group nan device_id_persons
    # and data that comes from 'uwb_sensors' into their own "groups" or buckets
    split_track_row_ids.extend(df_person_tray_distances_aggregated[
                               (df_person_tray_distances_aggregated['device_id_person'].isnull()) |
                               (df_person_tray_distances_aggregated['track_type_person'] == 'uwb_sensor')
                               ].index.values)

    # Filter step-2 (B):
    # Step B is to look for pose tracks that are diconnected but related and try to
    # group those together
    for _, carry_event_row in df_carry_events_with_track_ids.iterrows():
        tray_track_id = carry_event_row['tray_track_id']
        pose_tracks_filter_idx = \
            (df_person_tray_distances_aggregated['device_id_person'].notnull()) & \
            (df_person_tray_distances_aggregated['track_type_person'] == 'pose_track') & \
            (df_person_tray_distances_aggregated['tray_track_id'] == tray_track_id)

        df_person_tray_distances_aggregated_pose_tracks_only = df_person_tray_distances_aggregated[
            pose_tracks_filter_idx]
        for person_id in pd.unique(df_person_tray_distances_aggregated_pose_tracks_only['device_id_person']):
            if person_id == float(np.nan):
                continue

            df_person_tracks_filter_idx = df_person_tray_distances_aggregated_pose_tracks_only[
                'device_id_person'] == person_id
            split_track_row_ids.append(
                list(df_person_tray_distances_aggregated_pose_tracks_only[df_person_tracks_filter_idx].index.values))

    # Filter step-2 (C):
    # Use the split_track_row_ids to build and summarize "complete" tracks
    # If the previous split track time's total time is > 90% of the event time, retain the tracks
    filtered_tracks = []

    for row_id_groups in split_track_row_ids:
        if not isinstance(row_id_groups, list):
            row_id_groups = [row_id_groups]

        df_person_tracks = df_person_tray_distances_aggregated[
            df_person_tray_distances_aggregated.index.isin(row_id_groups)].copy()
        tray_track_length_in_seconds = df_carry_events_with_track_ids[
            df_carry_events_with_track_ids['tray_track_id'] == df_person_tracks.iloc[0]['tray_track_id']
        ].iloc[0]['track_length_seconds']

        if tray_track_length_in_seconds > 0 and (
                df_person_tracks['person_track_length_seconds'].sum() / tray_track_length_in_seconds) >= 0.9:
            df_person_tracks['track_length_percentage'] = df_person_tracks['person_track_length_seconds'] / \
                df_person_tracks['person_track_length_seconds'].sum()
            df_person_tracks['person_tray_distance_median_weighted'] = df_person_tracks['person_tray_distance_median'] * \
                df_person_tracks['track_length_percentage']

            df_person_flattened = df_person_tracks[['tray_track_id',
                                                    'device_id_person',
                                                    'person_id_person',
                                                    'person_name_person',
                                                    'person_short_name_person',
                                                    'person_anonymized_name_person',
                                                    'person_anonymized_short_name_person',
                                                    'device_id_tray',
                                                    'material_name_tray']].iloc[0].to_frame().transpose()
            df_person_flattened['person_tray_distance_median'] = df_person_tracks['person_tray_distance_median_weighted'].sum()
            df_person_flattened['person_track_length_seconds'] = df_person_tracks['person_track_length_seconds'].sum()

            df_person_grouped = df_person_tracks.groupby(
                [
                    'tray_track_id']).agg(
                {
                    'person_tray_distance_min': [
                        'min'
                    ],
                    'person_tray_distance_max': [
                        'max'
                    ]
                })

            df_person_grouped.columns = df_person_grouped.columns.to_flat_index()
            dataframe_tuple_columns_to_underscores(df_person_grouped, inplace=True)

            df_person_flattened['person_tray_distance_min'] = df_person_grouped['person_tray_distance_min_min'].iloc[0]
            df_person_flattened['person_tray_distance_max'] = df_person_grouped['person_tray_distance_max_max'].iloc[0]

            filtered_tracks.append(df_person_flattened)

    df_filtered_tracks = pd.concat(filtered_tracks)

    df_carry_events_distances_from_people = df_filtered_tracks \
        .merge(
            df_carry_events_with_track_ids[['tray_track_id', 'start', 'end']],
            how='left',
            on='tray_track_id')
    df_carry_events_distances_from_people.index.name = 'person_tray_track_id'

    return df_carry_events_distances_from_people


def infer_tray_device_interactions(df_features, df_carry_events, df_tray_centroids):
    """
    :param df_features:
    index,device_id,entity_type,tray_id,tray_name,material_assignment_id,material_id,material_name,person_id,person_name,quality,x_position,y_position,z_position,track_id,track_type
    2021-04-20 14:00:00.1000000 00:00,57f27842-2dca-4e96-ad06-c43de1ce5b5b,Tray,1ab9153d-a4bf-4b01-ae6f-64b162ebf178,GB Tray 1,91b1f7e7-e52f-4a7d-bace-02d945bf553b,8fd144de-4b64-4e26-a1aa-acdc39e08ca1,Spooning,<NA>,<NA>,8630.651339367998,-0.02254,6.97481,3.00000,57f27842-2dca-4e96-ad06-c43de1ce5b5b,uwb_sensor
    2021-04-20 14:00:00.2000000 00:00,57f27842-2dca-4e96-ad06-c43de1ce5b5b,Tray,1ab9153d-a4bf-4b01-ae6f-64b162ebf178,GB Tray 1,91b1f7e7-e52f-4a7d-bace-02d945bf553b,8fd144de-4b64-4e26-a1aa-acdc39e08ca1,Spooning,<NA>,<NA>,8726.985720758657,-0.01811,6.96696,3.00000,57f27842-2dca-4e96-ad06-c43de1ce5b5b,uwb_sensor
    2021-04-20 14:00:00.3000000 00:00,57f27842-2dca-4e96-ad06-c43de1ce5b5b,Tray,1ab9153d-a4bf-4b01-ae6f-64b162ebf178,GB Tray 1,91b1f7e7-e52f-4a7d-bace-02d945bf553b,8fd144de-4b64-4e26-a1aa-acdc39e08ca1,Spooning,<NA>,<NA>,8828.370805845896,-0.01419,6.95955,3.00000,57f27842-2dca-4e96-ad06-c43de1ce5b5b,uwb_sensor
    2021-04-20 14:00:00.4000000 00:00,57f27842-2dca-4e96-ad06-c43de1ce5b5b,Tray,1ab9153d-a4bf-4b01-ae6f-64b162ebf178,GB Tray 1,91b1f7e7-e52f-4a7d-bace-02d945bf553b,8fd144de-4b64-4e26-a1aa-acdc39e08ca1,Spooning,<NA>,<NA>,8733.670166453265,-0.01128,6.95299,3.00000,57f27842-2dca-4e96-ad06-c43de1ce5b5b,uwb_sensor
    ...
    2021-04-20 14:02:05.1000000 00:00,nan,Person,nan,nan,nan,nan,nan,nan,nan,nan,0.49127,1.88444,1.88444,0cfc7ea1ae9a45a1a17f598a5b1109de,pose_track
    2021-04-20 14:02:05.2000000 00:00,nan,Person,nan,nan,nan,nan,nan,nan,nan,nan,0.51071,1.90689,1.90689,0cfc7ea1ae9a45a1a17f598a5b1109de,pose_track
    2021-04-20 14:02:05.3000000 00:00,nan,Person,nan,nan,nan,nan,nan,nan,nan,nan,0.45057,1.94556,1.94556,0cfc7ea1ae9a45a1a17f598a5b1109de,pose_track
    2021-04-20 14:02:05.4000000 00:00,nan,Person,nan,nan,nan,nan,nan,nan,nan,nan,0.39042,1.98423,1.98423,0cfc7ea1ae9a45a1a17f598a5b1109de,pose_track
    ...
    2021-04-20 14:03:06.1000000 00:00,bbf3f7d4-994a-4418-93fc-ec0c009cc90a,Person,nan,nan,nan,nan,nan,62a0fd7a-e951-419b-a46a-2dd7b23136c4,Nynzie Noglo,nan,3.99970,5.47494,5.47494,8d6d4bec000e4168b371bfcbae4116f6,pose_track
    2021-04-20 14:03:06.2000000 00:00,bbf3f7d4-994a-4418-93fc-ec0c009cc90a,Person,nan,nan,nan,nan,nan,62a0fd7a-e951-419b-a46a-2dd7b23136c4,Nynzie Noglo,nan,3.98898,5.47557,5.47557,8d6d4bec000e4168b371bfcbae4116f6,pose_track
    2021-04-20 14:03:06.3000000 00:00,bbf3f7d4-994a-4418-93fc-ec0c009cc90a,Person,nan,nan,nan,nan,nan,62a0fd7a-e951-419b-a46a-2dd7b23136c4,Nynzie Noglo,nan,3.98487,5.47795,5.47795,8d6d4bec000e4168b371bfcbae4116f6,pose_track
    2021-04-20 14:03:06.4000000 00:00,bbf3f7d4-994a-4418-93fc-ec0c009cc90a,Person,nan,nan,nan,nan,nan,62a0fd7a-e951-419b-a46a-2dd7b23136c4,Nynzie Noglo,nan,3.98820,5.47480,5.47480,8d6d4bec000e4168b371bfcbae4116f6,pose_track



    :param df_carry_events:
    device_id,start,end,quality_median
    f37dd610-7335-4355-b706-0a1ea3f35519,2021-04-20 14:01:26.300000+00:00,2021-04-20 14:01:30.300000+00:00,8573.84593
    f37dd610-7335-4355-b706-0a1ea3f35519,2021-04-20 14:11:26.200000+00:00,2021-04-20 14:11:30.300000+00:00,8621.08054
    ...

    :param df_tray_centroids:
    x_centroid,y_centroid,device_id,start_datetime,end_datetime
    4.34030,5.12649,f37dd610-7335-4355-b706-0a1ea3f35519,2021-04-20 14:00:00.100000+00:00,2021-04-20 14:14:59.900000+00:00
    1.30512,11.29408,7fb13183-8f7b-4236-ad58-ddb37c571967,2021-04-20 14:00:00.100000+00:00,2021-04-20 14:14:59.900000+00:00
    -0.04017,6.96929,57f27842-2dca-4e96-ad06-c43de1ce5b5b,2021-04-20 14:00:00.100000+00:00,2021-04-20 14:14:59.900000+00:00
    3.78387,4.74192,5d17b2dc-9190-4495-bac0-dbfc60673b40,2021-04-20 14:00:00.100000+00:00,2021-04-20 14:14:59.900000+00:00
    3.91644,6.82516,6b643522-3490-4cea-a955-fce29509334c,2021-04-20 14:00:00.100000+00:00,2021-04-20 14:14:59.900000+00:00
    ...

    :return: Dataframe
    """
    df_carry_events_with_track_ids = modify_carry_events_with_track_ids(df_carry_events)
    df_filtered_people, df_filtered_trays_with_track_ids = filter_features_by_carry_events_and_split_by_device_type(
        df_features, df_carry_events_with_track_ids)
    # TODO: Filter 3d->2d tracks by time

    ###############
    # Determine nearest tray/person distances
    ###############

    # Build a dataframe with distances between all people and trays across all carry events times
    # TODO: Add 3d to 2d track as some form of people so we can compute distances to each of these tracks as well...
    df_person_tray_distances = generate_person_tray_distances(df_filtered_people, df_filtered_trays_with_track_ids)
    df_carry_events_distances_from_people = aggregate_clean_filter_person_tray_distances(
        df_person_tray_distances, df_carry_events_with_track_ids)

    #############
    # Determine trays positions at the start/end moments of tray carry
    #############
    df_positions_for_carry_event_moments = get_estimated_tray_location_from_carry_events(
        df_features, df_carry_events_with_track_ids)

    #############
    # Combine carry events w/ tray positions dataframe with the tray centroids dataframe
    #############
    df_carry_event_position_and_centroid = df_positions_for_carry_event_moments.rename_axis(
        'timestamp').reset_index().merge(df_tray_centroids, left_on='device_id', right_on='device_id')

    filter_centroids_with_start_end_time_match = (
        (df_carry_event_position_and_centroid['start_datetime'] < df_carry_event_position_and_centroid['timestamp']) &
        (df_carry_event_position_and_centroid['end_datetime'] > df_carry_event_position_and_centroid['timestamp']))

    df_carry_event_position_and_centroid = df_carry_event_position_and_centroid.loc[
        filter_centroids_with_start_end_time_match]
    df_carry_event_position_and_centroid.drop(
        labels=['start_datetime', 'end_datetime'],
        axis=1,
        inplace=True)

    #############
    # Calculate distance between tray positions and tray centroids/shelf-positions
    #############
    centroid_to_tray_location_distances = []
    centroid_to_tray_location_distances_columns = ['timestamp', 'device_id', 'tray_track_id', 'distance']
    centroid_cols = map_column_name_to_dimension_space('centroid', DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE)
    position_cols = map_column_name_to_dimension_space(
        'position', DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE)

    for idx, row in df_carry_event_position_and_centroid.iterrows():
        centroid_to_tray_location_distances.append(pd.DataFrame([[
            row['timestamp'],
            row['device_id'],
            row['tray_track_id'],
            np.linalg.norm(
                row[centroid_cols].to_numpy() - row[position_cols].to_numpy())
        ]], columns=centroid_to_tray_location_distances_columns))

    # pd.DataFrame(columns = column_names)
    if len(centroid_to_tray_location_distances) > 0:
        df_device_distance_from_source = pd.concat(centroid_to_tray_location_distances)
    else:
        df_device_distance_from_source = pd.DataFrame(columns=centroid_to_tray_location_distances_columns)

    #############
    # Merge tray centroid <-> position distance into a cleaned up carry events dataframe
    # containing tray_start_distance_from_source and tray_end_distance_from_source
    #############
    df_final_carry_events_with_start_distance_only = df_carry_events_with_track_ids \
        .merge(
            df_device_distance_from_source,
            how='left',
            left_on=['start', 'device_id', 'tray_track_id'],
            right_on=['timestamp', 'device_id', 'tray_track_id']) \
        .drop(
            ['timestamp'],
            axis=1) \
        .rename(
            columns={"distance": "tray_start_distance_from_source"})
    df_final_carry_events_with_distances = df_final_carry_events_with_start_distance_only \
        .merge(
            df_device_distance_from_source,
            how='left',
            left_on=['end', 'device_id', 'tray_track_id'],
            right_on=['timestamp', 'device_id', 'tray_track_id']) \
        .drop(
            ['timestamp'],
            axis=1) \
        .rename(
            columns={"distance": "tray_end_distance_from_source"})
    df_final_carry_events_with_distances.set_index('tray_track_id')

    #############
    # Perform final cleanup and filtering
    #############

    # Add detailed tray & material info the dataframe
    df_tray_features = df_features[df_features['entity_type'] == 'Tray']
    df_tray_assignments = df_tray_features.groupby(['device_id',
                                                    'tray_id',
                                                    'tray_name',
                                                    'material_assignment_id',
                                                    'material_id',
                                                    'material_name']).size().reset_index().drop(0,
                                                                                                1)
    df_final_carry_events_with_distances = df_final_carry_events_with_distances.merge(
        df_tray_assignments, how='left', left_on='device_id', right_on='device_id')

    # Find nearest person and filter out instances where tray and person are too far apart
    df_min_person_tray_track_ids = df_carry_events_distances_from_people.groupby(
        ['tray_track_id'])['person_tray_distance_median'].idxmin().rename("person_tray_track_id").to_frame()
    df_nearest_person_to_each_track = df_carry_events_distances_from_people[(
        (df_carry_events_distances_from_people.index.isin(df_min_person_tray_track_ids['person_tray_track_id'].tolist())) &
        (df_carry_events_distances_from_people['person_tray_distance_median']
         < CARRY_EVENT_DISTANCE_BETWEEN_TRAY_AND_PERSON)
    )]

    df_tray_interactions_pre_filter = df_final_carry_events_with_distances.merge(
        df_nearest_person_to_each_track, how='left')

    # # Determine each person's activity at the start and end of the carry track
    # # Append that activity to the tray interactions dataframe that is being constructed
    # df_nearest_persons_human_activity_at_start = pd.merge(df_features[['device_id', 'human_activity_category']].reset_index(),
    #                                                       df_tray_interactions_pre_filter[[
    #                                                           'tray_track_id', 'device_id_person', 'start']][df_tray_interactions_pre_filter['device_id_person'].notnull()],
    #                                                       how='right',
    #                                                       left_on=['index', 'device_id'],
    #                                                       right_on=['start', 'device_id_person'])
    # df_nearest_persons_human_activity_at_end = pd.merge(df_features[['device_id', 'human_activity_category']].reset_index(),
    #                                                     df_tray_interactions_pre_filter[[
    #                                                         'tray_track_id', 'device_id_person', 'end']][df_tray_interactions_pre_filter['device_id_person'].notnull()],
    #                                                     how='right',
    #                                                     left_on=['index', 'device_id'],
    #                                                     right_on=['end', 'device_id_person'])
    #
    # df_tray_interactions_pre_filter = df_tray_interactions_pre_filter.merge(df_nearest_persons_human_activity_at_start[['tray_track_id', 'human_activity_category']],
    #                                                                         how='left',
    #                                                                         on='tray_track_id').rename(columns={'human_activity_category': 'human_activity_category_start'})
    #
    # df_tray_interactions_pre_filter = df_tray_interactions_pre_filter.merge(df_nearest_persons_human_activity_at_end[['tray_track_id', 'human_activity_category']],
    #                                                                         how='left',
    # on='tray_track_id').rename(columns={'human_activity_category':
    # 'human_activity_category_end'})

    # Apply a filter that will retain instances where tray distance from
    # source/shelf is within min distance
    # (CARRY_EVENT_DISTANCE_BETWEEN_TRAY_AND_SHELF)
    # filter_trays_within_min_distance_from_source = (
    #     (df_tray_interactions_pre_filter['tray_start_distance_from_source'] < CARRY_EVENT_DISTANCE_BETWEEN_TRAY_AND_SHELF) |
    #     (df_tray_interactions_pre_filter['tray_end_distance_from_source'] < CARRY_EVENT_DISTANCE_BETWEEN_TRAY_AND_SHELF))
    # df_tray_interactions_pre_filter.loc[filter_trays_within_min_distance_from_source]
    df_tray_interactions = df_tray_interactions_pre_filter.copy()

    # Final dataframe contains:
    #   tray_device_id (str)
    #   start (date)
    #   end (date)
    #   person_device_id (str)
    #   person_name (str)
    #   tray_id (str)
    #   tray_name (str)
    #   material_assignment_id (str)
    #   material_id (str)
    #   material_name (str)
    #   person_tray_distance_median (float)
    #   devices_distance_max (float)
    #   devices_distance_min (float)
    #   tray_start_distance_from_source (float)
    #   tray_end_distance_from_source (float)
    #   interaction_type (str)
    #   human_activity_category_start (str)
    #   human_activity_category_end (str)
    logger.info("Tray motion interactions\n{}".format(df_tray_interactions))
    df_tray_interactions = df_tray_interactions\
        .rename(
            columns={
                'device_id': 'tray_device_id',
                'device_id_person': 'person_device_id',
                'person_id_person': 'person_id',
                'person_name_person': 'person_name',
                'person_short_name_person': 'person_short_name',
                'person_anonymized_name_person': 'person_anonymized_name',
                'person_anonymized_short_name_person': 'person_anonymized_short_name',
            })\
        .drop(
            labels=['tray_track_id', 'material_name_tray', 'device_id_tray'],
            axis=1)

    interaction_types = []
    for _, row in df_tray_interactions.iterrows():
        if row['tray_start_distance_from_source'] < CARRY_EVENT_DISTANCE_BETWEEN_TRAY_AND_SHELF and row['tray_end_distance_from_source'] < CARRY_EVENT_DISTANCE_BETWEEN_TRAY_AND_SHELF:
            interaction_types.append(InteractionType.CARRYING_FROM_AND_TO_SHELF.name)
        elif row['tray_start_distance_from_source'] < CARRY_EVENT_DISTANCE_BETWEEN_TRAY_AND_SHELF:
            interaction_types.append(InteractionType.CARRYING_FROM_SHELF.name)
        elif row['tray_end_distance_from_source'] < CARRY_EVENT_DISTANCE_BETWEEN_TRAY_AND_SHELF:
            interaction_types.append(InteractionType.CARRYING_TO_SHELF.name)
        else:
            interaction_types.append(InteractionType.CARRYING_BETWEEN_NON_SHELF_LOCATIONS.name)

    df_tray_interactions = df_tray_interactions.assign(interaction_type=interaction_types)
    return df_tray_interactions
