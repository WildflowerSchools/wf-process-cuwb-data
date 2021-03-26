import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier as skRandomForestClassifier
import sklearn.model_selection
from sklearn.preprocessing import StandardScaler

from process_cuwb_data.utils.log import logger


class UWBRandomForestClassifier:
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
            self.__classifier = skRandomForestClassifier(
                **self.attrs()
            )
        return self.__classifier

    @classifier.setter
    def classifier(self, classifier):
        self.__classifier = classifier

    def train_test_split(self, df_groundtruth, test_size=0.3, feature_field_names=[],
                         ground_truth_label_field_name='ground_truth_state'):
        X_all = df_groundtruth[feature_field_names].values
        y_all = df_groundtruth[ground_truth_label_field_name].values

        X_all_train, X_all_test, y_all_train, y_all_test = sklearn.model_selection.train_test_split(
            X_all,
            y_all,
            test_size=test_size,
            stratify=y_all
        )

        return X_all_train, X_all_test, y_all_train, y_all_test

    def tune(self,
             df_groundtruth,
             ground_truth_label_field_name='ground_truth_state',
             test_size=0.3,
             scale_features=False,
             param_grid=None,
             *args,
             **kwargs):
        if param_grid is None:
            param_grid = {
                'n_estimators': [75, 100, 200],
                'max_features': ['auto'],
                'max_depth': [None, 30, 50, 60],
                'criterion': ['gini', 'entropy'],
                'min_samples_split': [2, 5]
            }

        df_groundtruth[ground_truth_label_field_name] = df_groundtruth[ground_truth_label_field_name].str.lower()

        X_all_train, X_all_test, y_all_train, y_all_test = self.train_test_split(df_groundtruth, test_size)

        if scale_features:
            sc = StandardScaler()
            X_all_train = sc.fit_transform(X_all_train)

        rfc = self.classifier
        cv_rfc = sklearn.model_selection.GridSearchCV(estimator=rfc, param_grid=param_grid, n_jobs=-1, verbose=3)
        cv_rfc.fit(X_all_train, y_all_train)
        return cv_rfc

    def fit(self, df_groundtruth,
            feature_field_names=[],
            ground_truth_label_field_name='ground_truth_state',
            test_size=0.3,
            scale_features=False,
            *args,
            **kwargs):
        if not isinstance(self.classifier, skRandomForestClassifier):
            raise Exception("Classifier model type is {}, must be RandomForestClassifier".format(type(self.classifier)))

        df_groundtruth[ground_truth_label_field_name] = df_groundtruth[ground_truth_label_field_name].str.lower()

        X_all_train, X_all_test, y_all_train, y_all_test = self.train_test_split(
            df_groundtruth,
            feature_field_names=feature_field_names,
            ground_truth_label_field_name=ground_truth_label_field_name,
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
            logger.warn("RandomForestClassifier passed an empty feature set")
            return None

        if model is None:
            logger.error("RandomForestClassifier must be generated via training or supplied at init time")
            raise Exception("RandomForestClassifier required, is None")

        if not isinstance(model, skRandomForestClassifier):
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
