import numpy as np
import pandas as pd
from sklearn import cluster
from sklearn.mixture import GaussianMixture

from process_cuwb_data.utils.log import logger
from process_cuwb_data.uwb_motion_enum_carry_categories import CarryCategory
from .uwb_motion_classifier_tray_carry import TrayCarryClassifier

DIMENSIONS_WHEN_COMPUTING_CHILD_TRAY_DISTANCE = 2
DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE = 2


def map_column_name_to_dimension_space(column_name, num_dimensions):
    dims = ['x', 'y', 'z']
    return list(map(lambda d: "{}_{}".format(d, column_name), dims[0:num_dimensions]))


def validate_tray_centroids_dataframe(df_tray_centroids):
    if 'z_centroid' not in df_tray_centroids.columns:
        df_tray_centroids['z_centroid'] = 0.5

    required_columns = ['device_id', 'start_datetime', 'end_datetime', 'x_centroid', 'y_centroid', 'z_centroid']

    # Verify required columns exist
    missing_columns = []
    for rcolumn in required_columns:
        if rcolumn not in df_tray_centroids.columns:
            missing_columns.append(rcolumn)

    if len(missing_columns) > 0:
        return False, "Tray centroids data missing column(s) {}".format(missing_columns)

    if df_tray_centroids.x_centroid.dtype != 'float64' or \
            df_tray_centroids.y_centroid.dtype != 'float64' or \
            df_tray_centroids.z_centroid.dtype != 'float64':
        msg = "Invalid tray centroid datatype, position columns (x/y/z_centroid) should be float"
        return False, msg

    return True, ""


def classifier_filter_no_movement_from_tray_features(model, scaler, df_tray_features):
    tc = TrayCarryClassifier(model=model, feature_scaler=scaler)
    df_features_with_predictions = tc.predict(df_tray_features)

    motionless_mask = df_features_with_predictions['predicted_tray_carry_label'] == CarryCategory.NOT_CARRIED.name
    df_tray_features_no_movement = df_tray_features[motionless_mask].copy()
    return df_tray_features_no_movement


def heuristic_filter_no_movement_from_tray_features(df_tray_features):
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

    df_tray_features_no_movement = df_tray_movement_features_rounded[motionless_mask].copy()
    return df_tray_features_no_movement


def predict_tray_centroids(df_tray_features_no_movement):
    """
    Predict all tray's predominant resting position (shelf position)

    :param df_tray_features:
    :return: Dataframe with tray centroid positions and device_id
    """

    position_cols = map_column_name_to_dimension_space(
        'position_smoothed', DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE)
    centroid_cols = map_column_name_to_dimension_space('centroid', DIMENSIONS_WHEN_COMPUTING_TRAY_SHELF_DISTANCE)

    ###################
    # Use MeanShift to estimate the number of no-movement clusters for each tray
    ###################
    logger.info("Estimating # of clusters for each device")
    tray_clusters = []
    for device_id in pd.unique(df_tray_features_no_movement['device_id']):
        df_tray_no_movement_for_device = df_tray_features_no_movement[df_tray_features_no_movement['device_id'] == device_id].copy(
        )

        X = df_tray_no_movement_for_device[position_cols].copy().round(2)
        # Estimate the number of clusters per device, allow all processors to work
        logger.info("Estimating # of clusters for: {}".format(device_id))
        bandwidth = cluster.estimate_bandwidth(X, quantile=0.3, n_samples=15000)

        # If min_bin_freq is too large, MeanShift will fail
        min_bin_freq = 20
        if 15000 > len(X) >= 10000:
            min_bin_freq = 10
        elif len(X) < 10000:
            min_bin_freq = 2
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

    if tray_clusters == []:
        return None

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
        df_tray_no_movement_for_device = df_tray_features_no_movement[df_tray_features_no_movement['device_id'] == device_id]
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
            start_datetime=df_tray_features_no_movement[df_tray_features_no_movement['device_id'] == device_id].index.min(
            ),
            end_datetime=df_tray_features_no_movement[df_tray_features_no_movement['device_id'] == device_id].index.max())

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
