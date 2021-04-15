import multiprocessing
import pandas as pd

from process_cuwb_data.uwb_motion_classifier_random_forest import UWBRandomForestClassifier
from .utils.log import logger
from .uwb_motion_enum_carry_categories import CarryCategory
from .uwb_motion_filters import SmoothLabelsFilter, TrayCarryHmmFilter

DEFAULT_FEATURE_FIELD_NAMES = (
    'quality',
    'velocity_vector_magnitude',
    'velocity_vector_magnitude_mean',
    'velocity_vector_magnitude_stddev',
    'velocity_vector_magnitude_xy',
    'velocity_vector_magnitude_mean_xy',
    'velocity_vector_magnitude_stddev_xy',
    #
    # Not using add'l velocity features, add'l attributes didn't show improved results
    #
    # 'velocity_vector_magnitude_skew_xy',
    # 'velocity_vector_magnitude_variance_xy',
    # 'velocity_vector_magnitude_kurtosis_xy',
    # 'x_velocity_smoothed_magnitude',
    # 'y_velocity_smoothed_magnitude',
    # 'z_velocity_smoothed_magnitude',
    # 'x_velocity_smoothed',
    # 'y_velocity_smoothed',
    # 'z_velocity_smoothed',
    # 'x_velocity_mean',
    # 'y_velocity_mean',
    # 'z_velocity_mean',
    # 'velocity_average_mean',
    # 'x_velocity_stddev',
    # 'y_velocity_stddev',
    # 'z_velocity_stddev',
    # 'velocity_average_stddev',
    # 'x_velocity_skew',
    # 'y_velocity_skew',
    # 'z_velocity_skew',
    # 'velocity_average_skew',
    # 'x_velocity_variance',
    # 'y_velocity_variance',
    # 'z_velocity_variance',
    # 'velocity_average_variance',
    # 'x_velocity_kurtosis',
    # 'y_velocity_kurtosis',
    # 'z_velocity_kurtosis',
    # 'velocity_average_kurtosis',
    'x_y_velocity_correlation',
    # 'x_z_velocity_correlation',
    # 'y_z_velocity_correlation',
    # 'x_velocity_correlation_sum',
    # 'y_velocity_correlation_sum',
    # 'z_velocity_correlation_sum',
    'acceleration_vector_magnitude',
    'acceleration_vector_magnitude_mean',
    'acceleration_vector_magnitude_sum',
    'acceleration_vector_magnitude_min',
    'acceleration_vector_magnitude_max',
    'acceleration_vector_magnitude_stddev',
    'acceleration_vector_magnitude_skew',
    'acceleration_vector_magnitude_variance',
    'acceleration_vector_magnitude_kurtosis',
    'acceleration_vector_magnitude_energy',
    'x_acceleration_normalized',
    'y_acceleration_normalized',
    'z_acceleration_normalized',
    'x_acceleration_mean',
    'y_acceleration_mean',
    'z_acceleration_mean',
    'acceleration_average_mean',
    'x_acceleration_sum',
    'y_acceleration_sum',
    'z_acceleration_sum',
    'acceleration_average_sum',
    'x_acceleration_min',
    'y_acceleration_min',
    'z_acceleration_min',
    'acceleration_average_min',
    'x_acceleration_max',
    'y_acceleration_max',
    'z_acceleration_max',
    'acceleration_average_max',
    'x_acceleration_stddev',
    'y_acceleration_stddev',
    'z_acceleration_stddev',
    'acceleration_average_stddev',
    'x_acceleration_skew',
    'y_acceleration_skew',
    'z_acceleration_skew',
    'acceleration_average_skew',
    'x_acceleration_variance',
    'y_acceleration_variance',
    'z_acceleration_variance',
    'acceleration_average_variance',
    'x_acceleration_kurtosis',
    'y_acceleration_kurtosis',
    'z_acceleration_kurtosis',
    'acceleration_average_kurtosis',
    'x_acceleration_energy',
    'y_acceleration_energy',
    'z_acceleration_energy',
    'acceleration_average_energy',
    'x_y_acceleration_correlation',
    'x_z_acceleration_correlation',
    'y_z_acceleration_correlation',
    'x_acceleration_correlation_sum',
    'y_acceleration_correlation_sum',
    'z_acceleration_correlation_sum'
)


class TrayCarryClassifier(UWBRandomForestClassifier):
    def __init__(self, model=None, feature_scaler=None, feature_field_names=DEFAULT_FEATURE_FIELD_NAMES,
                 ground_truth_label_field_name='ground_truth_state',
                 prediction_field_name='predicted_tray_carry_label'):
        super().__init__(n_estimators=100, max_depth=30, max_features='auto',
                         min_samples_leaf=1, min_samples_split=2,
                         class_weight='balanced_subsample', criterion='entropy', verbose=0, n_jobs=-1)

        self.model = model
        self.scaler = feature_scaler
        self.feature_field_names = list(feature_field_names)
        self.ground_truth_label_field_name = ground_truth_label_field_name
        self.prediction_field_name = prediction_field_name

    def tune(self, df_groundtruth,
             test_size=0.3,
             scale_features=False,
             param_grid=None,
             *args,
             **kwargs):
        # We create another instance of the wfRandomForestClassifier to be sure the classifier is fresh
        rfc = UWBRandomForestClassifier(verbose=1)
        cv_rfc = rfc.tune(
            df_groundtruth,
            prediction_field_name=self.prediction_field_name,
            test_size=test_size,
            scale_features=scale_features,
            param_grid=param_grid)

        logger.info("Ideal tuned params: {}".format(cv_rfc.best_params_))

        return cv_rfc

    def fit(self, df_groundtruth,
            test_size=0.3,
            scale_features=True,
            classifier=None,
            *args,
            **kwargs):
        if classifier is not None:
            self.classifier = classifier

        result = super().fit(df_groundtruth=df_groundtruth,
                             feature_field_names=self.feature_field_names,
                             ground_truth_label_field_name=self.ground_truth_label_field_name,
                             test_size=test_size,
                             scale_features=scale_features)

        self.model = result['model']
        self.scaler = result['scaler']

        return result

    def filter_and_smooth_predictions(self, device_id, df_device_features, window=10):
        logger.info("Filter tray carry classification anomalies (hmm model) for device ID {}".format(device_id))
        df_device_features = TrayCarryHmmFilter().filter(df_device_features, self.prediction_field_name)

        logger.info("Smooth tray carry classification for device ID {}".format(device_id))
        df_device_features = SmoothLabelsFilter(
            window=window).filter(
            df_predictions=df_device_features,
            prediction_column_name=self.prediction_field_name)

        logger.info(
            "Carry Prediction (post filter and smoothing) for device ID {}:\n{}".format(
                device_id,
                df_device_features.groupby(self.prediction_field_name).size()))

        return (device_id, df_device_features)

    def predict(self, df_features,
                *args,
                **kwargs):
        df_features = df_features.copy()

        predictions = super().predict(
            df_features=df_features,
            model=self.model,
            scaler=self.scaler,
            feature_field_names=self.feature_field_names)
        if predictions is None:
            return None
        df_features[self.prediction_field_name] = predictions

        logger.info(
            "Carry Prediction (pre filter and smoothing):\n{}".format(
                df_features.groupby(self.prediction_field_name).size()))

        # Convert carry state from string to int
        carry_category_mapping_dictionary = CarryCategory.as_name_id_dict()  # Category label lookup is case insensitive
        df_features[self.prediction_field_name] = df_features[self.prediction_field_name].map(
            lambda x: carry_category_mapping_dictionary[x])

        # HMM Model and smoothing can take awhile, use multiprocessing to help speed things up
        p = multiprocessing.Pool()

        # df_dict = dict()
        results = []
        for device_id in pd.unique(df_features['device_id']):
            df_device_features = df_features.loc[df_features['device_id'] == device_id].copy().sort_index()
            results.append(
                p.apply_async(
                    self.filter_and_smooth_predictions,
                    kwds=dict(
                        device_id=device_id,
                        df_device_features=df_device_features)))

        df_dict = dict(p.get() for p in results)
        p.close()

        df_features = pd.concat(df_dict.values())

        # Convert carry state from int to string
        df_features[self.prediction_field_name] = df_features[self.prediction_field_name].map(
            CarryCategory.as_id_name_dict())

        logger.info(
            "Carry Prediction (post filter and smoothing):\n{}".format(
                df_features.groupby(self.prediction_field_name).size()))

        return df_features
