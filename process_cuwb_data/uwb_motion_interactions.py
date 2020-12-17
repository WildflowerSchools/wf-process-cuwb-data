from functools import partial, reduce
import multiprocessing
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn import cluster
from sklearn.mixture import GaussianMixture
import sys
import time

from .log import logger
from .util import dataframe_tuple_columns_to_underscores
from .uwb_motion_interaction_types import InteractionType


# TODO: Ignoring z-axis when computing distance for now, reconsider after further testing CUWB anchors
DIMENSIONS_WHEN_COMPUTING_CHILD_TRAY_DISTANCE = 2
DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE = 2


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

    :param df_carry_events:
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
        'position_smoothed', DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE)
    for _, row in df_carry_events_with_track_ids_and_augmented_times.iterrows():
        device_id_mask = (df_tray_features['device_id'] == row['device_id'])
        start_mask = (df_tray_features.index == row['start_augmented']) & device_id_mask
        end_mask = (df_tray_features.index == row['end_augmented']) & device_id_mask

        cols = [*['device_id'], *position_cols]
        start_position = df_tray_features.loc[start_mask][cols]
        start_position = start_position.assign(carry_moment='start')
        start_position = start_position.assign(tray_track_id=row['tray_track_id'])
        start_position.index = [row['start']]

        end_position = df_tray_features.loc[end_mask][cols]
        end_position = end_position.assign(carry_moment='end')
        end_position = end_position.assign(tray_track_id=row['tray_track_id'])
        end_position.index = [row['end']]

        carry_events_with_positions.append(start_position)
        carry_events_with_positions.append(end_position)

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
        'position_smoothed', DIMENSIONS_WHEN_COMPUTING_CHILD_TRAY_DISTANCE)

    df_people_and_trays = df_people_by_idx.join(df_trays_by_idx, how='inner', lsuffix='_person', rsuffix='_tray')
    distances = cdist(df_people_by_idx[position_cols].to_numpy(),
                      df_trays_by_idx[position_cols].to_numpy(),
                      metric='euclidean')

    return df_people_and_trays.assign(devices_distance=distances.flatten())


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

    df_child_tray_distances = pd.concat(
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

    return df_child_tray_distances


def predict_tray_centroids(df_tray_features):
    """
    Predict all tray's predominant resting position (shelf position)

    :param df_tray_features:
    :return: Dataframe with tray centroid positions in 3d space and device_id
    """
    position_cols = map_column_name_to_dimension_space(
        'position_smoothed', DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE)
    centroid_cols = map_column_name_to_dimension_space('centroid', DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE)

    df_tray_movement_features = df_tray_features[['device_id',
                                                  'x_position_smoothed',
                                                  'y_position_smoothed',
                                                  'z_position_smoothed',
                                                  'x_velocity_smoothed',
                                                  'y_velocity_smoothed',
                                                  'x_acceleration_normalized',
                                                  'y_acceleration_normalized',
                                                  'z_acceleration_normalized']]

    # Round off movement features before finding "no movement" instances
    df_tray_movement_features_rounded = df_tray_movement_features.copy()
    df_tray_movement_features_rounded[['x_velocity_smoothed',
                                       'y_velocity_smoothed',
                                       'x_acceleration_normalized',
                                       'y_acceleration_normalized',
                                       'z_acceleration_normalized']] = df_tray_movement_features[['x_velocity_smoothed',
                                                                                                  'y_velocity_smoothed',
                                                                                                  'x_acceleration_normalized',
                                                                                                  'y_acceleration_normalized',
                                                                                                  'z_acceleration_normalized']].round(2)

    # Round off tray movement to integers, this will help toward getting an estimate of the # of cluster locations
    df_generalized_tray_locations = df_tray_movement_features_rounded.round(0).groupby(
        [*['device_id'], *position_cols]).size().reset_index().rename(columns={0: 'count'})

    motionless_mask = (df_tray_movement_features_rounded['x_velocity_smoothed'] == 0.0) & \
                      (df_tray_movement_features_rounded['y_velocity_smoothed'] == 0.0) & \
                      (df_tray_movement_features_rounded['x_acceleration_normalized'] == 0.0) & \
                      (df_tray_movement_features_rounded['y_acceleration_normalized'] == 0.0) & \
                      (df_tray_movement_features_rounded['z_acceleration_normalized'] == 0.0)

    ###################
    # Use MeanShift to estimate the number of no-movement clusters for each tray
    ###################
    logger.info("Estimating # of clusters for each device")
    tray_clusters = []
    df_tray_no_movement = df_tray_movement_features_rounded[motionless_mask].copy()

    for device_id in pd.unique(df_tray_no_movement['device_id']):
        df_tray_no_movement_for_device = df_tray_no_movement[df_tray_no_movement['device_id'] == device_id].copy()

        X = df_tray_no_movement_for_device[position_cols].copy().round(2)
        # Estimate the number of clusters per device, allow all processors to work
        logger.info("Estimating # of clusters for: {}".format(device_id))
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3, n_samples=15000)
        # TODO: if min_bin_freq is too large, this will fail
        clustering = cluster.MeanShift(bandwidth=bandwidth, n_jobs=-1, bin_seeding=True, min_bin_freq=20).fit(X)
        logger.info("Clusters for device {} est: {}".format(device_id, len(clustering.cluster_centers_)))
        for label, val in enumerate(clustering.cluster_centers_):
            tray_clusters.append(pd.DataFrame([[device_id, val[0], val[1], np.count_nonzero(
                np.array(clustering.labels_ == label))]], columns=[*['device_id'], *centroid_cols, *['count']]))

    df_tray_clusters = pd.concat(tray_clusters)

    ###################
    # Filter no-movement tray clusters by locations making up more than 5% of the day
    ###################
    tray_clusters = []
    for device_id in pd.unique(df_tray_clusters['device_id']):
        df_tray_clusters_by_device = df_tray_clusters[df_tray_clusters['device_id'] == device_id].copy()
        df_tray_clusters_by_device['percent'] = df_tray_clusters_by_device['count'] / \
            df_tray_clusters_by_device['count'].sum()
        tray_cluster = df_tray_clusters_by_device[df_tray_clusters_by_device['percent'] > 0.05]
        tray_clusters.append(tray_cluster)

    df_tray_clusters = pd.concat(tray_clusters).reset_index(drop=True)

    ####################
    # Use GaussianMixture algorithm to predict highest occurring cluster centroid coordinates
    ####################
    logger.info("Estimating tray centroids (tray's shelf position)")
    tray_centroids = []
    for device_id in pd.unique(df_tray_clusters['device_id']):
        logger.info("Estimating tray centroids for device: {}".format(device_id))
        df_tray_no_movement_for_device = df_tray_no_movement[df_tray_no_movement['device_id'] == device_id]
        df_tray_clusters_for_device = df_tray_clusters[df_tray_clusters['device_id'] == device_id]

        model = GaussianMixture(n_components=len(df_tray_clusters_for_device))
        model.fit(df_tray_no_movement_for_device[position_cols])
        df_tray_centroid = df_tray_no_movement_for_device.assign(centroids=model.predict(
            df_tray_no_movement_for_device[position_cols]))

        centers = np.empty(shape=(model.n_components, DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE))
        centers[:] = np.NaN
        for ii in range(model.n_components):
            centers[ii, :] = model.means_[ii]

        df_cluster_centers = pd.DataFrame(centers, columns=centroid_cols)
        df_tray_centroid = df_tray_centroid.merge(df_cluster_centers, how='left', left_on='centroids', right_index=True)
        df_tray_centroid_grouped = df_tray_centroid.groupby(
            centroid_cols).size().reset_index().rename(columns={0: 'count'})

        max_idx = df_tray_centroid_grouped[['count']].idxmax()

        tray_centroids.append(df_tray_centroid_grouped.loc[max_idx][centroid_cols].assign(device_id=[device_id]))

    df_tray_centroids = pd.concat(tray_centroids).reset_index(drop=True)
    return df_tray_centroids


def extract_tray_device_interactions(df_features, df_carry_events):
    df_carry_events_with_track_ids = modify_carry_events_with_track_ids(df_carry_events)
    df_filtered_people, df_filtered_trays_with_track_ids = filter_features_by_carry_events_and_split_by_device_type(
        df_features, df_carry_events_with_track_ids)

    ###############
    # Determine nearest tray/person distances
    ###############
    df_person_tray_distances = generate_person_tray_distances(df_filtered_people, df_filtered_trays_with_track_ids)

    # Group and aggregate each person and tray-carry-track combination to get the mean distance during the carry event
    # FYI: Grouping creates a dataframe with multiple column axis
    df_child_tray_distances_aggregated = df_person_tray_distances.groupby(
        [
            'tray_track_id',
            'device_id_person',
            'person_name_person',
            'device_id_tray',
            'material_name_tray']).agg(
        {
            'devices_distance': [
                'count',
                'mean',
                'median',
                'min',
                'max']}).reset_index()  # .set_index('tray_track_id')
    df_child_tray_distances_aggregated.columns = df_child_tray_distances_aggregated.columns.to_flat_index()
    dataframe_tuple_columns_to_underscores(df_child_tray_distances_aggregated, inplace=True)

    # TODO: Filter by nearest tray <> person (or not, it could be possible two children carry a tray together)
    df_carry_events_distances_from_people = df_child_tray_distances_aggregated \
        .merge(
            df_carry_events_with_track_ids[['tray_track_id', 'start', 'end']],
            how='left',
            on='tray_track_id')

    #############
    # Determine tray centroids (this could be substituted with user defined values)
    #############
    df_tray_centroids = predict_tray_centroids(df_features[df_features['entity_type'] == 'Tray'])
    df_positions_for_carry_event_moments = get_estimated_tray_location_from_carry_events(
        df_features, df_carry_events_with_track_ids)

    # Create dataframe that holds estimated carry event positions with each tray's centroid
    df_carry_event_position_and_centroid = df_positions_for_carry_event_moments.rename_axis(
        'timestamp').reset_index().merge(df_tray_centroids, left_on='device_id', right_on='device_id')

    centroid_to_tray_location_distances = []
    centroid_cols = map_column_name_to_dimension_space('centroid', DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE)
    position_cols = map_column_name_to_dimension_space(
        'position_smoothed', DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE)

    for idx, row in df_carry_event_position_and_centroid.iterrows():
        centroid_to_tray_location_distances.append(pd.DataFrame([[
            row['timestamp'],
            row['device_id'],
            row['tray_track_id'],
            np.linalg.norm(
                row[centroid_cols].to_numpy() - row[position_cols].to_numpy())
        ]], columns=['timestamp', 'device_id', 'tray_track_id', 'distance']))
    df_device_distance_from_source = pd.concat(centroid_to_tray_location_distances)

    #############
    # Use tray centroids to compute start/end distances from tray source/shelf
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

    # Filter out instances where tray and person are too far apart
    df_grouped_carry_events_distances_from_people = df_carry_events_distances_from_people.groupby(['tray_track_id'])
    filter_grouped_carry_events_distances = df_grouped_carry_events_distances_from_people.agg(
        {'devices_distance_median': 'min'})['devices_distance_median'] < 1.25
    df_nearest_person_to_each_track = df_carry_events_distances_from_people.loc[filter_grouped_carry_events_distances]
    df_tray_interactions_pre_filter = df_final_carry_events_with_distances.merge(
        df_nearest_person_to_each_track, how='left').drop(['device_id'], 1)

    # Filter out instances where tray distance from source/shelf is too far apart
    filter_trays_within_min_distance_from_source = (
        (df_tray_interactions_pre_filter['tray_start_distance_from_source'] < 1.25) |
        (df_tray_interactions_pre_filter['tray_end_distance_from_source'] < 1.25))
    df_tray_interactions = df_tray_interactions_pre_filter.loc[filter_trays_within_min_distance_from_source]

    # Final dataframe contains:
    #   start (date)
    #   end (date)
    #   person_device_id (str)
    #   person_name (str)
    #   tray_device_id (str)
    #   tray_device_name (str)
    #   tray_name (str)
    #   tray_material_name (str)
    #   material_id (str)
    #   material_name (str)
    #   devices_distance_median (float)
    #   devices_distance_max (float)
    #   devices_distance_min (float)
    #   devices_distance_average (float)
    #   devices_distance_count (float)
    #   tray_start_distance_from_source (float)
    #   tray_end_distance_from_source (float)
    logger.info("Tray motion interactions\n{}".format(df_tray_interactions))
    df_tray_interactions = df_tray_interactions.rename(
        columns={
            'device_id_person': 'person_device_id',
            'person_name_person': 'person_name'})
    df_tray_interactions = df_tray_interactions.drop(
        labels=['device_id_tray', 'material_name_tray'],
        axis=1)

    interaction_types = []
    for _, row in df_tray_interactions.iterrows():
        if row['tray_start_distance_from_source'] < 1.25:
            interaction_types.append(InteractionType.CARRYING_FROM_SHELF.name)
        elif row['tray_end_distance_from_source'] < 1.25:
            interaction_types.append(InteractionType.CARRYING_TO_SHELF.name)
        else:
            interaction_types.append(InteractionType.CARRYING_TO_SHELF.unknown)

    df_tray_interactions = df_tray_interactions.assign(interaction_type=interaction_types)
    return df_tray_interactions
