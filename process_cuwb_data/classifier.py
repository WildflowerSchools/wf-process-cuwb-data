import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler

from .log import logger


DEFAULT_FEATURE_FIELD_NAMES = [
    'x_velocity_smoothed',
    'y_velocity_smoothed',
    'x_acceleration_normalized',
    'y_acceleration_normalized',
    'z_acceleration_normalized',
]


class DefaultRFClassifier:
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


class Train:
    def __init__(self, groundtruth_features, feature_field_names=None,
                 target_field_name='ground_truth_state', classifier=DefaultRFClassifier().classifier):
        self.groundtruth_features = groundtruth_features

        if feature_field_names is None:
            feature_field_names = DEFAULT_FEATURE_FIELD_NAMES
        self.feature_field_names = feature_field_names
        self.target_field_name = target_field_name

        if not isinstance(classifier, RandomForestClassifier):
            raise Exception("Classifier model type is {}, must be RandomForestClassifier".format(type(model)))
        self.classifier = classifier

    def train(self, test_size=0.2, scale_features=True):
        X_all = self.groundtruth_features[self.feature_field_names].values
        y_all = self.groundtruth_features[self.target_field_name].values

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
            confusion_matric_plot=disp
        )


class Inference:
    def __init__(self, model=None, scaler=None, feature_field_names=None,):
        if model is None:
            raise Exception("Classifier model required, is None")

        if not isinstance(model, RandomForestClassifier):
            raise Exception("Classifier model type is {}, must be RandomForestClassifier".format(type(model)))

        if scaler is not None and not isinstance(scaler, StandardScaler):
            raise Exception("Feature scaler type is {}, must be StandardScaler".format(type(scaler)))

        if feature_field_names is None:
            feature_field_names = DEFAULT_FEATURE_FIELD_NAMES
        self.feature_field_names = feature_field_names

        self.model = model
        self.scaler = scaler

    def infer(self, features):
        features = features.copy()

        classifier_features = features[self.feature_field_names]

        if self.scaler is not None:
            classifier_features = self.scaler.transform(classifier_features)

        features['predicted_state'] = self.model.predict(classifier_features)

        logger.info("Prediction:\n{}".format(features.groupby('predicted_state').size()))

        return features
