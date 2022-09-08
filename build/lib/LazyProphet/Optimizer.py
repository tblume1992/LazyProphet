# -*- coding: utf-8 -*-
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import time
import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


class Optimize:
    def __init__(self,
                 y,
                 lazyprophet_class,
                 seasonal_period=0,
                 n_folds=3,
                 test_size=None,
                 n_trials=100):
        self.y = y
        self.lazyprophet_class = lazyprophet_class
        if isinstance(seasonal_period, list):
            self.max_pulse = max(seasonal_period)
        else:
            self.max_pulse = seasonal_period
        self.seasonal_period = seasonal_period
        self.n_folds = n_folds
        self.test_size = test_size
        self.n_trials = n_trials

    def logic_layer(self):
        n_samples = len(self.y)
        test_size = n_samples//(self.n_folds + 1)
        if n_samples - test_size < self.max_pulse:
            self.seasonal_period = 0

    def scorer(self, model_obj, y, metric, cv):
        cv_splits = cv.split(y)
        mses = []
        for train_index, test_index in cv_splits:
            try:
                model_obj.fit(y[train_index])
                predicted = model_obj.predict(len(y[test_index]))
                mses.append(mean_squared_error(y[test_index], predicted))
            except:
                mses.append(np.inf)
        return np.mean(mses)


    def objective(self, trial):
        params = {
            "n_estimators": trial.suggest_int(name="n_estimators", low=25, high=500),
            "num_leaves": trial.suggest_int("num_leaves", 8, 128),
            "n_basis": trial.suggest_int("n_basis", 0, 15),
            "decay": trial.suggest_categorical("decay", ['auto',
                                                         .05,
                                                         .1,
                                                         .25,
                                                         .5,
                                                         .75,
                                                         .9,
                                                         .99]),
        }
        if self.seasonal_period:
            params.update({'seasonal_period': trial.suggest_categorical("seasonal_period", [None, self.seasonal_period])})
            params.update({'ar': trial.suggest_int(name="ar", low=0, high=self.max_pulse)})
        else:
            params.update({'ar': trial.suggest_int(name="ar", low=0, high=4)})
        params['ar'] = list(range(1, 1 + params['ar']))
        clf = self.lazyprophet_class(**params)
        score = self.scorer(clf, self.y, mean_squared_error, TimeSeriesSplit(self.n_folds, test_size=self.test_size))
        return score

    def fit(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        return study




