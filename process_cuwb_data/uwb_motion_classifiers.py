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
    # 'quality', - Ignoring quality in classifier for now, filtering by quality median value post-categorization instead
    'x_velocity_smoothed',
    'y_velocity_smoothed',
    'x_acceleration_normalized',
    'y_acceleration_normalized',
    'z_acceleration_normalized'
]


class TrayMotionRandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=60, max_features='auto',
                 min_samples_leaf=1, min_samples_split=2,
                 class_weight='balanced_subsample', criterion='entropy', verbose=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight
        self.criterion = criterion
        self.verbose = verbose

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
            verbose=self.verbose
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

    def train_test_split(self, df_groundtruth, test_size=0.2):
        X_all = df_groundtruth[self.feature_field_names].values
        y_all = df_groundtruth[self.prediction_field_name].values

        X_all_train, X_all_test, y_all_train, y_all_test = sklearn.model_selection.train_test_split(
            X_all,
            y_all,
            test_size=test_size,
            stratify=y_all
        )

        return X_all_train, X_all_test, y_all_train, y_all_test

    def tune(self, df_groundtruth, test_size=0.2, scale_features=True, param_grid=None):
        if param_grid is None:
            param_grid = {
                'n_estimators': [75, 100, 200],
                'max_features': ['auto'],
                'max_depth': [None, 30, 50, 60],
                'criterion': ['gini', 'entropy'],
                'min_samples_split': [2, 5]
            }

        X_all_train, X_all_test, y_all_train, y_all_test = self.train_test_split(df_groundtruth, test_size)

        if scale_features:
            sc = StandardScaler()
            X_all_train = sc.fit_transform(X_all_train)

        rfc = TrayMotionRandomForestClassifier(verbose=3).classifier
        cv_rfc = sklearn.model_selection.GridSearchCV(estimator=rfc, param_grid=param_grid, n_jobs=-1, verbose=3)
        cv_rfc.fit(X_all_train, y_all_train)

        logger.info("Ideal tuned params: {}".format(cv_rfc.best_params_))

    def train(self, df_groundtruth, test_size=0.2, scale_features=True,
              classifier=TrayMotionRandomForestClassifier().classifier):
        if not isinstance(classifier, RandomForestClassifier):
            raise Exception("Classifier model type is {}, must be RandomForestClassifier".format(type(classifier)))

        X_all_train, X_all_test, y_all_train, y_all_test = self.train_test_split(df_groundtruth, test_size)

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
        """
        Smooth out predicted instances of carry/not-carry changes that don't occur within a stable
        rolling window of carry events. Stable rolling windows are when the number of frames (windows)
        are uniform.

        :param df: Dataframe to smooth
        :param prediction_column_name: Predicted carry column
        :param window: Number of frames that must be uniform to be considered a "stable" event period
        :param inplace: Modify dataframe in-place
        :return: Modified dataframe (if inplace is False)
        """
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
            raise Exception(
                "TrayCarryClassifier model type is {}, must be RandomForestClassifier".format(
                    type(
                        self.model)))

        if self.scaler is not None and not isinstance(self.scaler, StandardScaler):
            raise Exception("Feature scaler type is {}, must be StandardScaler".format(type(self.scaler)))

        df_features = df_features.copy()

        classifier_features = df_features[self.feature_field_names]

        if self.scaler is not None:
            classifier_features = self.scaler.transform(classifier_features)

        df_features['predicted_state'] = self.model.predict(classifier_features)

        logger.info(
            "Carry Prediction (pre filter and smoothing):\n{}".format(
                df_features.groupby('predicted_state').size()))

        # Convert state from string to int
        df_features['predicted_state'] = df_features['predicted_state'].map(CarryCategory.as_name_id_dict())

        df_dict = dict()
        for device_id in pd.unique(df_features['device_id']):
            df_device_features = df_features.loc[df_features['device_id'] == device_id].copy().sort_index()

            logger.info("Filter tray carry classification anomalies for device ID {}".format(device_id))
            df_device_features = TrayCarryHmmFilter().filter(df_device_features, 'predicted_state')

            logger.info("Smooth tray carry classification for device ID {}".format(device_id))
            df_device_features = self.inference_post_filter_smooth_predictions(df_device_features, 'predicted_state')

            df_dict[device_id] = df_device_features

        df_features = pd.concat(df_dict.values())
        # Convert state from int to string
        df_features['predicted_state'] = df_features['predicted_state'].map(CarryCategory.as_id_name_dict())

        logger.info(
            "Carry Prediction (post filter and smoothing):\n{}".format(
                df_features.groupby('predicted_state').size()))

        return df_features
