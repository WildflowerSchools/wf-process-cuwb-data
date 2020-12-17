import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler

from .uwb_motion_carry_categories import CarryCategory
from .log import logger
from .uwb_motion_filters import TrayCarryHmmFilter

DEFAULT_FEATURE_FIELD_NAMES = [
    'x_velocity_smoothed',
    'y_velocity_smoothed',
    'x_acceleration_normalized',
    'y_acceleration_normalized',
    'z_acceleration_normalized',
]


class TrayMotionRandomForestClassifier:
    def __init__(self, n_estimators=75, max_depth=30, min_samples_leaf=1,
                 min_samples_split=5, class_weight='balanced_subsample'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight

        self.__classifier = None

    def attrs(self):
        return dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight
        )

    @property
    def classifier(self):
        if self.__classifier is None:
            self.__classifier = RandomForestClassifier(
                **self.attrs()
            )
        return self.__classifier


class TrayCarryClassifier:
    def __init__(self, model=None, feature_scaler=None, feature_field_names=DEFAULT_FEATURE_FIELD_NAMES,
                 prediction_field_name='ground_truth_state'):
        self.model = model
        self.scaler = feature_scaler
        self.feature_field_names = feature_field_names
        self.prediction_field_name = prediction_field_name

    def train(self, df_groundtruth, test_size=0.2, scale_features=True,
              classifier=TrayMotionRandomForestClassifier().classifier):
        if not isinstance(classifier, RandomForestClassifier):
            raise Exception("Classifier model type is {}, must be RandomForestClassifier".format(type(classifier)))

        X_all = df_groundtruth[self.feature_field_names].values
        y_all = df_groundtruth[self.prediction_field_name].values

        X_all_train, X_all_test, y_all_train, y_all_test = sklearn.model_selection.train_test_split(
            X_all,
            y_all,
            test_size=test_size,
            stratify=y_all
        )

        values_train, counts_train = np.unique(y_all_train, return_counts=True)
        values_test, counts_test = np.unique(y_all_test, return_counts=True)

        logger.info("Training label balance:\n{}".format(dict(zip(values_train, counts_train / np.sum(counts_train)))))
        logger.info("Test label balance:\n{}".format(dict(zip(values_test, counts_test / np.sum(counts_test)))))

        sc = None
        if scale_features:
            sc = StandardScaler()
            X_all_train = sc.fit_transform(X_all_train)
            X_all_test = sc.transform(X_all_test)

        classifier.fit(X_all_train, y_all_train)

        logger.info(
            "Confusion Matrix:\n{}".format(
                sklearn.metrics.confusion_matrix(
                    y_all_test,
                    classifier.predict(X_all_test))))

        logger.info('Classification Report:\n{}'.format(
            sklearn.metrics.classification_report(
                y_all_test,
                classifier.predict(X_all_test))))

        disp = sklearn.metrics.plot_confusion_matrix(
            classifier,
            X_all_test,
            y_all_test,
            cmap=plt.cm.Blues,
            normalize=None,
            values_format='n'
        )
        disp.ax_.set_title(
            'Random forest ({} estimators, {} max depth): test set'.format(
                classifier.n_estimators,
                classifier.max_depth))

        self.model = classifier
        self.scaler = sc

        return dict(
            model=classifier,
            scaler=sc,
            confusion_matrix_plot=disp
        )

    def inference_post_filter_smooth_predictions(self, df, prediction_column_name, window=10, inplace=False):
        if not inplace:
            df = df.copy()

        rolling_window = df[prediction_column_name].rolling(window=window, center=True)
        carry_stability = (rolling_window.min() == rolling_window.max())

        last_prediction = None
        for ii_idx, (time_idx, row) in enumerate(df.iterrows()):
            row_carry_stability = carry_stability.loc[time_idx]
            if ii_idx == 0:
                last_prediction = row[prediction_column_name]
                continue

            if row[prediction_column_name] != last_prediction:
                if row_carry_stability == False:
                    df.loc[time_idx, prediction_column_name] = int(not bool(row[prediction_column_name]))

            if row_carry_stability == True:
                last_prediction = row[prediction_column_name]

        if not inplace:
            return df

    def inference(self, df_features):
        if df_features is None or len(df_features) == 0:
            return None

        if self.model is None:
            logger.error("TrayCarryClassifier model must generated via training or supplied at init time")
            raise Exception("TrayCarryClassifier model required, is None")

        if not isinstance(self.model, RandomForestClassifier):
            raise Exception("TrayCarryClassifier model type is {}, must be RandomForestClassifier".format(type(self.model)))

        if self.scaler is not None and not isinstance(self.scaler, StandardScaler):
            raise Exception("Feature scaler type is {}, must be StandardScaler".format(type(self.scaler)))

        df_features = df_features.copy()

        classifier_features = df_features[self.feature_field_names]

        if self.scaler is not None:
            classifier_features = self.scaler.transform(classifier_features)

        df_features['predicted_state'] = self.model.predict(classifier_features)

        logger.info("Carry Prediction:\n{}".format(df_features.groupby('predicted_state').size()))

        # Convert state from string to int
        df_features['predicted_state'] = df_features['predicted_state'].map(CarryCategory.as_name_id_dict())

        df_dict = dict()
        for device_id in pd.unique(df_features['device_id']):
            df_device_features = df_features.loc[df_features['device_id'] == device_id].copy().sort_index()

            logger.info("Filter tray carry prediction anomalies for device ID {}".format(device_id))
            df_device_features = TrayCarryHmmFilter().filter(df_device_features, 'predicted_state')

            logger.info("Smooth tray carry predictions for device ID {}".format(device_id))
            df_device_features = self.inference_post_filter_smooth_predictions(df_device_features, 'predicted_state')

            df_dict[device_id] = df_device_features

        df_features = pd.concat(df_dict.values())
        # Convert state from int to string
        df_features['predicted_state'] = df_features['predicted_state'].map(CarryCategory.as_id_name_dict())
        return df_features
