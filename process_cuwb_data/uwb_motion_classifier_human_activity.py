import tensorflow as tf


import sklearn.model_selection
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURE_FIELD_NAMES = (
    'quality',
    'velocity_vector_magnitude',
    'velocity_vector_magnitude_mean',
    'velocity_vector_magnitude_stddev',
    'velocity_vector_magnitude_xy',
    'velocity_vector_magnitude_mean_xy',
    'velocity_vector_magnitude_stddev_xy',
    'velocity_vector_magnitude_skew_xy',
    'velocity_vector_magnitude_variance_xy',
    'velocity_vector_magnitude_kurtosis_xy',
    # 'x_velocity_smoothed_magnitude',
    # 'y_velocity_smoothed_magnitude',
    'z_velocity_smoothed_magnitude',
    # 'x_velocity_smoothed',
    # 'y_velocity_smoothed',
    'z_velocity_smoothed',
    # 'x_velocity_mean',
    # 'y_velocity_mean',
    'z_velocity_mean',
    'velocity_average_mean',
    # 'x_velocity_stddev',
    # 'y_velocity_stddev',
    'z_velocity_stddev',
    'velocity_average_stddev',
    # 'x_velocity_skew',
    # 'y_velocity_skew',
    'z_velocity_skew',
    'velocity_average_skew',
    # 'x_velocity_variance',
    # 'y_velocity_variance',
    'z_velocity_variance',
    'velocity_average_variance',
    # 'x_velocity_kurtosis',
    # 'y_velocity_kurtosis',
    'z_velocity_kurtosis',
    'velocity_average_kurtosis',
    # 'x_y_velocity_correlation',
    # 'x_z_velocity_correlation',
    'y_z_velocity_correlation',
    # 'x_velocity_correlation_sum',
    # 'y_velocity_correlation_sum',
    'z_velocity_correlation_sum',
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
    # 'x_acceleration_normalized',
    # 'y_acceleration_normalized',
    'z_acceleration_normalized',
    # 'x_acceleration_mean',
    # 'y_acceleration_mean',
    'z_acceleration_mean',
    'acceleration_average_mean',
    # 'x_acceleration_sum',
    # 'y_acceleration_sum',
    'z_acceleration_sum',
    'acceleration_average_sum',
    # 'x_acceleration_min',
    # 'y_acceleration_min',
    'z_acceleration_min',
    'acceleration_average_min',
    # 'x_acceleration_max',
    # 'y_acceleration_max',
    'z_acceleration_max',
    'acceleration_average_max',
    # 'x_acceleration_stddev',
    # 'y_acceleration_stddev',
    'z_acceleration_stddev',
    'acceleration_average_stddev',
    # 'x_acceleration_skew',
    # 'y_acceleration_skew',
    'z_acceleration_skew',
    'acceleration_average_skew',
    # 'x_acceleration_variance',
    # 'y_acceleration_variance',
    'z_acceleration_variance',
    'acceleration_average_variance',
    # 'x_acceleration_kurtosis',
    # 'y_acceleration_kurtosis',
    'z_acceleration_kurtosis',
    'acceleration_average_kurtosis',
    # 'x_acceleration_energy',
    # 'y_acceleration_energy',
    'z_acceleration_energy',
    'acceleration_average_energy',
    # 'x_y_acceleration_correlation',
    # 'x_z_acceleration_correlation',
    # 'y_z_acceleration_correlation',
    # 'x_acceleration_correlation_sum',
    # 'y_acceleration_correlation_sum',
    'z_acceleration_correlation_sum'
)


class LSTM:
    def __init__(self, attrs):
        super(self)

    def fit(self):
        pass

    def predict(self):
        pass


class HumanActivityLSTM:
    def __init__(self):
        super(self)
        self.__classifier = None

    def attrs(self):
        return dict()

    @property
    def classifier(self):
        if self.__classifier is None:
            self.__classifier = LSTM(
                **self.attrs()
            )
        return self.__classifier


class HumanActivityClassifier:
    def __init__(self, model=None, feature_scaler=None, feature_field_names=DEFAULT_FEATURE_FIELD_NAMES,
                 prediction_field_name='ground_truth_state'):
        self.model = model
        self.scaler = feature_scaler
        self.feature_field_names = list(feature_field_names)
        self.prediction_field_name = prediction_field_name

    def train_test_split(self, df_groundtruth, test_size=0.3):
        X_all = df_groundtruth[self.feature_field_names].values
        y_all = df_groundtruth[self.prediction_field_name].values

        X_all_train, X_all_test, y_all_train, y_all_test = sklearn.model_selection.train_test_split(
            X_all,
            y_all,
            test_size=test_size,
            stratify=y_all
        )

        return X_all_train, X_all_test, y_all_train, y_all_test

    def train(self, df_groundtruth, test_size=0.3, scale_features=True,
              classifier=HumanActivityLSTM().classifier):
        if not isinstance(classifier, HumanActivityLSTM):
            raise Exception("Classifier model type is {}, must be HumanActivityLSTM".format(type(classifier)))