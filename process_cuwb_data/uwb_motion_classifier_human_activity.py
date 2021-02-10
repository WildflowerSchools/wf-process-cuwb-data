import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler
# import tensorflow as tf

from .uwb_motion_enum_human_activities import HumanActivity
from .uwb_motion_filters import SmoothLabelsFilter
from .utils.log import logger

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


class WildflowerRandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=30, max_features='auto',
                 min_samples_leaf=1, min_samples_split=2,
                 class_weight='balanced_subsample', criterion='entropy', verbose=0, n_jobs=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.criterion = criterion
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.__classifier = None

    def attrs(self):
        return dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            criterion=self.criterion,
            verbose=self.verbose,
            n_jobs=self.n_jobs
        )

    @property
    def classifier(self):
        if self.__classifier is None:
            self.__classifier = RandomForestClassifier(
                **self.attrs()
            )
        return self.__classifier

    @classifier.setter
    def classifier(self, classifier):
        self.__classifier = classifier

    def train_test_split(self, df_groundtruth, test_size=0.3, feature_field_names=[],
                         prediction_field_name='ground_truth_state'):
        X_all = df_groundtruth[feature_field_names].values
        y_all = df_groundtruth[prediction_field_name].values

        X_all_train, X_all_test, y_all_train, y_all_test = sklearn.model_selection.train_test_split(
            X_all,
            y_all,
            test_size=test_size,
            stratify=y_all
        )

        return X_all_train, X_all_test, y_all_train, y_all_test

    def fit(self, df_groundtruth,
            feature_field_names=[],
            prediction_field_name='ground_truth_state',
            test_size=0.3,
            scale_features=True,
            *args,
            **kwargs):
        if not isinstance(self.classifier, RandomForestClassifier):
            raise Exception("Classifier model type is {}, must be RandomForestClassifier".format(type(self.classifier)))

        df_groundtruth[prediction_field_name] = df_groundtruth[prediction_field_name].str.lower()

        X_all_train, X_all_test, y_all_train, y_all_test = self.train_test_split(
            df_groundtruth,
            feature_field_names=feature_field_names,
            prediction_field_name=prediction_field_name,
            test_size=test_size)

        values_train, counts_train = np.unique(y_all_train, return_counts=True)
        values_test, counts_test = np.unique(y_all_test, return_counts=True)

        logger.info("Training label balance:\n{}".format(dict(zip(values_train, counts_train / np.sum(counts_train)))))
        logger.info("Test label balance:\n{}".format(dict(zip(values_test, counts_test / np.sum(counts_test)))))

        sc = None
        if scale_features:
            sc = StandardScaler()
            X_all_train = sc.fit_transform(X_all_train)
            X_all_test = sc.transform(X_all_test)

        self.classifier.fit(X_all_train, y_all_train)

        logger.info(
            "Confusion Matrix:\n{}".format(
                sklearn.metrics.confusion_matrix(
                    y_all_test,
                    self.classifier.predict(X_all_test))))

        logger.info('Classification Report:\n{}'.format(
            sklearn.metrics.classification_report(
                y_all_test,
                self.classifier.predict(X_all_test))))

        disp = sklearn.metrics.plot_confusion_matrix(
            self.classifier,
            X_all_test,
            y_all_test,
            cmap=plt.cm.Blues,
            normalize=None,
            values_format='n'
        )
        disp.ax_.set_title(
            'Random forest ({} estimators, {} max depth): test set'.format(
                self.classifier.n_estimators,
                self.classifier.max_depth))

        return dict(
            model=self.classifier,
            scaler=sc,
            confusion_matrix_plot=disp
        )

    def predict(self, df_features,
                model,
                scaler=None,
                feature_field_names=[],
                *args,
                **kwargs):
        if df_features is None or len(df_features) == 0:
            return None

        if model is None:
            logger.error("RandomForestClassifier must be generated via training or supplied at init time")
            raise Exception("RandomForestClassifier required, is None")

        if not isinstance(model, RandomForestClassifier):
            raise Exception(
                "RandomForestClassifier model type is {}, must be RandomForestClassifier".format(
                    type(
                        model)))

        if scaler is not None and not isinstance(scaler, StandardScaler):
            raise Exception("Feature scaler type is {}, must be StandardScaler".format(type(self.scaler)))

        df_features = df_features.copy()

        classifier_features = df_features[feature_field_names]

        if scaler is not None:
            classifier_features = scaler.transform(classifier_features)

        return model.predict(classifier_features)


class HumanActivityClassifier(WildflowerRandomForestClassifier):
    def __init__(self, model=None, feature_scaler=None,
                 feature_field_names=DEFAULT_FEATURE_FIELD_NAMES,
                 prediction_field_name='ground_truth_state'):
        super().__init__(n_estimators=100, max_depth=30, max_features='auto',
                         min_samples_leaf=1, min_samples_split=2,
                         class_weight='balanced_subsample', criterion='entropy', verbose=0, n_jobs=-1)
        self.model = model
        self.scaler = feature_scaler
        self.feature_field_names = list(feature_field_names)
        self.prediction_field_name = prediction_field_name

    def inference_filter_and_smooth_predictions(self, device_id, df_device_features, window=5):
        #logger.info("Filter tray carry classification anomalies (hmm model) for device ID {}".format(device_id))
        #df_device_features = TrayCarryHmmFilter().filter(df_device_features, self.prediction_field_name)

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
                             prediction_field_name=self.prediction_field_name,
                             test_size=test_size,
                             scale_features=scale_features)

        self.model = result['model']
        self.scaler = result['scaler']

        return result

    def predict(self, df_features,
                *args,
                **kwargs):
        predictions = super().predict(
            df_features=df_features,
            model=self.model,
            scaler=self.scaler,
            feature_field_names=self.feature_field_names)
        df_features[self.prediction_field_name] = predictions

        logger.info(
            "Activity Prediction (pre filter and smoothing):\n{}".format(
                df_features.groupby(self.prediction_field_name).size()))

        # Convert carry state from string to int
        human_activity_mapping_dictionary = HumanActivity.as_name_id_dict()  # Category label lookup is case insensitive
        df_features[self.prediction_field_name] = df_features[self.prediction_field_name].map(
            lambda x: human_activity_mapping_dictionary[x])

        # Filtering and smoothing can take awhile, use multiprocessing to help speed things up
        p = multiprocessing.Pool()

        # df_dict = dict()
        results = []
        for device_id in pd.unique(df_features['device_id']):
            df_device_features = df_features.loc[df_features['device_id'] == device_id].copy().sort_index()
            results.append(
                p.apply_async(
                    self.inference_filter_and_smooth_predictions,
                    kwds=dict(
                        device_id=device_id,
                        df_device_features=df_device_features)))

        df_dict = dict(p.get() for p in results)
        p.close()

        df_features = pd.concat(df_dict.values())

        # Convert carry state from int to string
        df_features[self.prediction_field_name] = df_features[self.prediction_field_name].map(
            HumanActivity.as_id_name_dict())

        logger.info(
            "Carry Prediction (post filter and smoothing):\n{}".format(
                df_features.groupby(self.prediction_field_name).size()))

        return df_features
