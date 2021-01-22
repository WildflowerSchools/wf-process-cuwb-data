from functools import partial
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
CARRY_EVENT_DISTANCE_BETWEEN_TRAY_AND_PERSON = 1.25
CARRY_EVENT_DISTANCE_BETWEEN_TRAY_AND_SHELF = 1.25


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


def validate_tray_centroids_dataframe(df_tray_centroids):
    # Ignoring start_time and end_time for now
    required_columns = ['device_id', 'start_datetime', 'end_datetime', 'x_centroid', 'y_centroid', 'z_centroid']

    # Verify required columns exist
    missing_columns = []
    for rcolumn in required_columns:
        if rcolumn not in df_tray_centroids.columns:
            missing_columns.append(rcolumn)

    if len(missing_columns) > 0:
        return False, "Tray centroids data missing column(s) {}".format(missing_columns)

    if df_tray_centroids.x_centroid.dtype != 'float64' or\
            df_tray_centroids.y_centroid.dtype != 'float64' or\
            df_tray_centroids.z_centroid.dtype != 'float64':
        msg = "Invalid tray centroid datatype, position columns (x/y/z_centroid) should be float"
        return False, msg

    return True, ""


def get_estimated_tray_location_from_carry_events(df_features, df_carry_events):
    df_tray_features = df_features[df_features['entity_type'] == 'Tray']

    # Fudge start/stop times to get a better guess at resting tray locations
    df_carry_events_with_track_ids_and_augmented_times = augment_carry_events_start_and_end_times(df_carry_events)

    carry_events_with_positions = []
    position_cols = map_column_name_to_dimension_space(
        'position_smoothed', DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE)
    for _, row in df_carry_events_with_track_ids_and_augmented_times.iterrows():
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
    :return: Dataframe with tray centroid positions and device_id
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
                                                                                                  'z_acceleration_normalized']].round(1)

    # Round off tray movement to integers, this will help toward getting an estimate of the # of cluster locations
    motionless_mask = (df_tray_movement_features_rounded['x_velocity_smoothed'] == 0.0) & \
                      (df_tray_movement_features_rounded['y_velocity_smoothed'] == 0.0) & \
                      (df_tray_movement_features_rounded['x_acceleration_normalized'] == 0.0) & \
                      (df_tray_movement_features_rounded['y_acceleration_normalized'] == 0.0)
    # (df_tray_movement_features_rounded['z_acceleration_normalized'] == 0.0)  # Ignoring z acceleration, seems to report erroneously

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
        min_bin_freq = 20
        if 15000 > len(X) >= 10000:
            min_bin_freq = 10
        elif len(X) < 10000:
            min_bin_freq = 2
        # TODO: if min_bin_freq is too large, this will fail
        clustering = cluster.MeanShift(
            bandwidth=bandwidth,
            n_jobs=-1,
            bin_seeding=True,
            min_bin_freq=min_bin_freq).fit(X)
        logger.info("Clusters for device {} est: {}".format(device_id, len(clustering.cluster_centers_)))
        for label, val in enumerate(clustering.cluster_centers_):
            tray_clusters.append(
                pd.DataFrame([[
                    device_id,
                    *val[0:DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE],
                    np.count_nonzero(
                        np.array(clustering.labels_ == label))]],
                    columns=[*['device_id'], *centroid_cols, *['count']]))

    df_tray_clusters = pd.concat(tray_clusters)

    ###################
    # Filter and retain no-movement tray clusters by locations making up more than 5% of the day
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

        # n_components is number of clusters that we should estimate
        model = GaussianMixture(n_components=len(df_tray_clusters_for_device))
        model.fit(df_tray_no_movement_for_device[position_cols])

        # predict the centroid index for each row/position in df_tray_no_movement_for_device
        df_tray_centroid = df_tray_no_movement_for_device.assign(centroids=model.predict(
            df_tray_no_movement_for_device[position_cols]))

        # capture the predicted centroids
        centers = np.empty(shape=(model.n_components, DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE))
        centers[:] = np.NaN
        for ii in range(model.n_components):
            centers[ii, :] = model.means_[ii]

        df_cluster_centers = pd.DataFrame(centers, columns=centroid_cols)

        # Join the position dataframe (df_tray_no_movement_for_device) with
        # centroid coords using the centroid index column
        df_tray_centroid = df_tray_centroid.merge(df_cluster_centers, how='left', left_on='centroids', right_index=True)
        df_tray_centroid_grouped = df_tray_centroid.groupby(
            centroid_cols).size().reset_index().rename(columns={0: 'count'})

        # Retain the highest occurring centroid
        # TODO: This logic could be improved. Where a tray resides most may not be it's primary shelf location
        #       Nor is it safe to assume a tray will have only a single shelf location during the day
        max_idx = df_tray_centroid_grouped[['count']].idxmax()

        df_tray_centroid = df_tray_centroid_grouped.loc[max_idx][centroid_cols]
        df_tray_centroid = df_tray_centroid.assign(
            device_id=[device_id],
            start_datetime=df_tray_features[df_tray_features['device_id'] == device_id].index.min(),
            end_datetime=df_tray_features[df_tray_features['device_id'] == device_id].index.max())

        tray_centroids.append(df_tray_centroid)

    df_tray_centroids = pd.concat(tray_centroids).reset_index(drop=True)

    # Output dataframe format (z_centroid added if DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE == 3):
    # idx            start_datetime                end_datetime x_centroid y_centroid  device_id
    # 0	  2020-01-17 13:00:00+00:00   2020-01-17 23:00:00+00:00   1.039953   7.852625  44fefd70-1790-4b8f-976c-58caf4d1d7e3
    # 1	  2020-01-17 13:00:00+00:00   2020-01-17 23:00:00+00:00   6.390389   8.804649  9a93a83f-e146-42e9-973a-673f228b75c9
    # 2	  2020-01-17 13:00:00+00:00   2020-01-17 23:00:00+00:00  -1.303979   9.555855  c7c8988c-0a25-45e8-b823-a85650366274
    # 3   2020-01-17 13:00:00+00:00   2020-01-17 23:00:00+00:00   2.328303
    # 1.865759  d9df153d-678d-4946-b78e-0c549c7d2156
    return df_tray_centroids


def extract_tray_device_interactions(df_features, df_carry_events, df_tray_centroids=None):
    df_carry_events_with_track_ids = modify_carry_events_with_track_ids(df_carry_events)
    df_filtered_people, df_filtered_trays_with_track_ids = filter_features_by_carry_events_and_split_by_device_type(
        df_features, df_carry_events_with_track_ids)

    ###############
    # Determine nearest tray/person distances
    ###############

    # Build a dataframe with distances between all people and trays across all carry events times
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
    df_carry_events_distances_from_people.index.name = 'person_tray_track_id'

    #############
    # Determine tray centroids (this could be substituted with user defined values)
    #############
    if df_tray_centroids is None:
        df_tray_centroids = predict_tray_centroids(df_features[df_features['entity_type'] == 'Tray'])

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
        'position_smoothed', DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE)

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

    #df_final_carry_events_with_distances['person_tray_track_id'] = pd.factorize(df_carry_events_distances_from_people['tray_track_id', 'person_id', 'device_id'])[0]

    # Filter out instances where tray and person are too far apart
    # df_grouped_carry_events_distances_from_people = df_carry_events_distances_from_people.groupby(['tray_track_id']).agg({'devices_distance_median': 'min'})
    # df_grouped_carry_events_distances_from_people.index.name = 'person_tray_id'

    df_min_person_tray_track_ids = df_carry_events_distances_from_people.groupby(
        ['tray_track_id'])['devices_distance_median'].idxmin().rename("person_tray_track_id").to_frame()
    df_nearest_person_to_each_track = df_carry_events_distances_from_people[(
        (df_carry_events_distances_from_people.index.isin(df_min_person_tray_track_ids['person_tray_track_id'].tolist())) &
        (df_carry_events_distances_from_people['devices_distance_median']
         < CARRY_EVENT_DISTANCE_BETWEEN_TRAY_AND_PERSON)
    )]

    # filter = (df_grouped_carry_events_distances_from_people['devices_distance_median'] < 1.25 &
    #           df_grouped_carry_events_distances_from_people.index.isin(df_min_distance_person_tray_ids['person_tray_id'].tolist()))

    #filter_grouped_carry_events_distances = df_grouped_carry_events_distances_from_people['devices_distance_median'] < 1.25
    #df_unique_track_ids = df_grouped_carry_events_distances_from_people[filter_grouped_carry_events_distances]

    #df_nearest_person_to_each_track = df_carry_events_distances_from_people[df_carry_events_distances_from_people['tray_track_id'].isin(df_unique_track_ids.index.tolist())]
    # df_nearest_person_to_each_track = df_grouped_carry_events_distances_from_people[filter]

    df_tray_interactions_pre_filter = df_final_carry_events_with_distances.merge(
        df_nearest_person_to_each_track, how='left')

    # Filter out instances where tray distance from source/shelf is too far apart
    filter_trays_within_min_distance_from_source = (
        (df_tray_interactions_pre_filter['tray_start_distance_from_source'] < CARRY_EVENT_DISTANCE_BETWEEN_TRAY_AND_SHELF) |
        (df_tray_interactions_pre_filter['tray_end_distance_from_source'] < CARRY_EVENT_DISTANCE_BETWEEN_TRAY_AND_SHELF))
    df_tray_interactions = df_tray_interactions_pre_filter.loc[filter_trays_within_min_distance_from_source]

    # Final dataframe contains:
    #   start (date)
    #   end (date)
    #   person_device_id (str)
    #   person_name (str)
    #   tray_id (str)
    #   tray_device_id (str)
    #   tray_name (str)
    #   material_assignment_id (str)
    #   material_id (str)
    #   material_name (str)
    #   devices_distance_median (float)
    #   devices_distance_max (float)
    #   devices_distance_min (float)
    #   devices_distance_average (float)
    #   devices_distance_count (float)
    #   tray_start_distance_from_source (float)
    #   tray_end_distance_from_source (float)
    #   interaction_type (str)
    logger.info("Tray motion interactions\n{}".format(df_tray_interactions))
    df_tray_interactions = df_tray_interactions.rename(
        columns={
            'device_id': 'tray_device_id',
            'device_id_person': 'person_device_id',
            'person_name_person': 'person_name'})
    df_tray_interactions = df_tray_interactions.drop(
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
            interaction_types.append(InteractionType.CARRYING_TO_SHELF.unknown)

    df_tray_interactions = df_tray_interactions.assign(interaction_type=interaction_types)
    return df_tray_interactions
