# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 08:19:32 2022

@author: Tyler Blume
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import optuna.integration.lightgbm as lgb
import optuna
from scipy import stats
import lightgbm as gbm
import warnings
from LazyProphet.LinearBasisFunction import LinearBasisFunction
from LazyProphet.FourierBasisFunction import FourierBasisFunction
warnings.filterwarnings("ignore")


class LazyProphet:
    
    def __init__(self,
                 objective='regression',
                 seasonal_period=None,
                 fourier_order=10,
                 n_basis=10,
                 ar=None,
                 ma_windows=None,
                 decay=None,
                 scale=True,
                 weighted=True,
                 decay_average=False,
                 seasonality_weights=None,
                 linear_trend=None,
                 boosting_params=None,
                 series_features=None,
                 return_proba=False):
        self.objective = objective
        self.exogenous = None
        if seasonal_period is not None:
            if not isinstance(seasonal_period, list):
                seasonal_period = [seasonal_period]
        self.seasonal_period = seasonal_period
        if objective == 'classification':
            scale = False
            linear_trend = False
        self.scale = scale
        if ar is not None:
            if not isinstance(ar, list):
                ar = [ar]
        self.ar = ar
        if ma_windows is not None:
            if not isinstance(ma_windows, list):
                ma_windows = [ma_windows]
        self.ma_windows = ma_windows
        self.fourier_order = fourier_order
        self.decay = decay
        self.n_basis = n_basis
        self.weighted = weighted
        self.series_features = series_features
        self.component_dict = {}
        self.decay_average = decay_average
        self.seasonality_weights = seasonality_weights
        self.linear_trend = linear_trend
        self.return_proba = return_proba
        if self.objective == 'regression':
            metric = 'rmse'
            objective = 'regression'
        elif self.objective == 'classification':
            metric = 'cross-entropy'
            objective = 'binary'
        if boosting_params is None:
            self.boosting_params = {
                                    "objective": objective,
                                    "metric": metric,
                                    "verbosity": -1,
                                    "boosting_type": "gbdt",
                                    "seed": 42,
                                    'linear_tree': False,
                                    'learning_rate': .15,
                                    'min_child_samples': 5,
                                    'num_leaves': 31,
                                    'num_iterations': 50
                                }
        else:
            self.boosting_params = boosting_params

    def linear_test(self, y):
        y = y.copy().reshape((-1,))
        xi = np.arange(1, len(y) + 1)
        xi = xi**2
        slope, intercept, r_value, p_value, std_err = stats.linregress(xi,y)
        trend_line = slope*xi*r_value + intercept
        if self.linear_trend is None or self.linear_trend == 'auto':
            n_bins = (1 + len(y)**(1/3) * 2)
            # n_bins = int(len(y) / 13)
            splitted_array = np.array_split(y.reshape(-1,), int(n_bins))
            mean_splits = np.array([np.mean(i) for i in splitted_array])
            grad = np.gradient(mean_splits)
            threshold = .9 * n_bins
            if sum(grad < 0) >= threshold or sum(grad > 0) >= threshold:
                growth = True
                # print('True')
            # asc_array = np.sort(mean_splits)
            # desc_array = np.flip(asc_array)
            # if all(asc_array == mean_splits):
            #     growth = True
            # elif all(desc_array == mean_splits):
            #     growth = True
            else:
                growth = False
            if (growth):
                self.linear_trend = True
            else:
                self.linear_trend = False
        self.slope = slope * r_value
        self.penalty = r_value
        self.intercept = intercept
        return trend_line

    def get_piecewise(self, y):
        self.lbf = LinearBasisFunction(n_changepoints=self.n_basis,
                                  decay=self.decay,
                                  weighted=self.weighted)
        basis = self.lbf.get_basis(y)
        return basis

    def get_harmonics(self, y, seasonal_period):
        self.fbf = FourierBasisFunction(self.fourier_order,
                                        self.seasonality_weights)
        basis = self.fbf.get_harmonics(y, seasonal_period)
        return basis

    @staticmethod
    def shift(xs, n):
        e = np.empty_like(xs)
        if n >= 0:
            e[:n] = np.nan
            e[n:] = xs[:-n]
        else:
            e[n:] = np.nan
            e[:n] = xs[-n:]
        return e

    @staticmethod
    def moving_average(y, window):
        y = pd.Series(y.reshape(-1,))
        ma = np.array(y.rolling(window).mean())
        return ma.reshape((-1, 1))

    def build_input(self, y, exogenous=None):
        X = np.arange(len(y))
        X = X.reshape((-1, 1))
        if self.n_basis is not None:
            if len(y) <= self.n_basis - 1:
                self.n_basis = len(y) - 1
            self.basis = self.get_piecewise(y)
            X = np.append(X, self.basis, axis=1)
            self.component_dict['basis'] = self.basis
        if self.seasonal_period:
            for period in self.seasonal_period:
                harmonics = self.get_harmonics(y, period)
                self.component_dict['harmonics ' + str(period)] = harmonics
                X = np.append(X, harmonics, axis=1)
        if self.exogenous is not None:
            X = np.append(X, exogenous, axis=1)
        if self.ar is not None:
            for ar_order in self.ar:
                shifted_y = self.scaled_y.copy()
                shifted_y = LazyProphet.shift(shifted_y, ar_order)
                X = np.append(X, shifted_y.reshape(-1, 1), axis=1)
        if self.ma_windows is not None:
            for ma_order in self.ma_windows:
                ma = LazyProphet.moving_average(self.scaled_y, ma_order)
                X = np.append(X, ma, axis=1)
        return X

    def scale_input(self, y):
        self.scaler = StandardScaler()
        self.scaler.fit(np.asarray(y).reshape(-1, 1))
        self.scaled_y = y.copy()
        self.scaled_y = self.scaler.transform(self.scaled_y.reshape(-1, 1))

    def fit(self, y, X=None):
        self.exogenous = X
        self.og_y = y
        if self.series_features is None:
            y = np.array(y)
        else:
            y = self.series_features
        if self.linear_trend is None or self.linear_trend:
            fitted_trend = self.linear_test(y)
        if self.linear_trend:
            y = np.subtract(y, fitted_trend)
        #TODO: Should we disable here?
        # if self.linear_trend:
        #     self.ar = None
        #     self.decay = None
        if self.scale:
            self.scale_input(y)
        else:
            self.scaled_y = y.copy()
        self.X = self.build_input(self.scaled_y)
        if self.objective == 'regression':
            self.model_obj = gbm.LGBMRegressor(**self.boosting_params)
        if self.objective == 'classification':
            self.model_obj = gbm.LGBMClassifier(**self.boosting_params)
        if self.series_features is None:
            self.model_obj.fit(self.X, self.scaled_y.reshape(-1, ))
        else:
            self.model_obj.fit(self.X, self.og_y.reshape(-1, ))
        #commented out from basic feature selection
        # self.columns = pd.Series(lp_model.model_obj.feature_importances_).sort_values().index[-100:]
        # self.model_obj.fit(self.X[:, self.columns], self.scaled_y.reshape(-1, ))
        if self.return_proba:
            fitted = self.model_obj.predict_proba(self.X)
            fitted = fitted[:, 1].reshape(-1,1)
        else:
            fitted = self.model_obj.predict(self.X).reshape(-1,1)
        if self.scale:
            fitted = self.scaler.inverse_transform(fitted)
        if self.linear_trend:
            fitted = np.add(fitted.reshape(-1,1), fitted_trend.reshape(-1,1))
        return fitted

    def recursive_predict(self, X, forecast_horizon):
        self.future_X = X
        #TODO: This is just...horrible
        predictions = []
        self.full = self.scaled_y.copy()
        if self.ar is not None:
            self.future_X = np.append(self.future_X,
                                      np.zeros((len(X), len(self.ar))),
                                      axis=1)
        if self.ma_windows is not None:
            self.future_X = np.append(self.future_X,
                                      np.zeros((len(X), len(self.ma_windows))),
                                      axis=1)
        for step in range(forecast_horizon):
            if self.ar is not None:
                for i, ar_order in enumerate(self.ar):
                    column_slice = -len(self.ar) + i
                    if step < ar_order:
                        self.future_X[step, column_slice] = self.scaled_y[-ar_order + step]
                    else:
                        self.future_X[step, column_slice] = predictions[-ar_order]
            if self.ma_windows is not None:
                for i, ma_window in enumerate(self.ma_windows):
                    column_slice = -len(self.ma_windows) + i
                    ma = np.mean(self.full[-ma_window:])
                    self.future_X[step, column_slice] = ma
            recursive_X = self.future_X[step, :].reshape(1, -1)
            if self.return_proba:
                predicted = self.model_obj.predict_proba(recursive_X)
                predicted = predicted[:, 1]
            else:
                predicted = self.model_obj.predict(recursive_X)
            predictions.append(predicted)
            self.full = np.append(self.full, predictions[-1])
        return np.array(predictions)

    def predict(self, forecast_horizon, future_X=None):
        X = np.arange(forecast_horizon) + len(self.scaled_y)
        X = X.reshape((-1, 1))
        if self.n_basis is not None:
            basis = self.lbf.get_future_basis(self.component_dict['basis'],
                                          forecast_horizon,
                                          average=self.decay_average)
            X = np.append(X, basis, axis=1)
        if self.seasonal_period:
            for period in self.seasonal_period:
                harmonics = self.component_dict['harmonics ' + str(period)]
                future_harmonics = self.fbf.get_future_harmonics(harmonics,
                                                                 forecast_horizon,
                                                                 period)
                X = np.append(X, future_harmonics, axis=1)
        if self.exogenous is not None:
            X = np.append(X, future_X, axis=1)
        if self.ar is not None or self.ma_windows is not None:
            predicted = self.recursive_predict(X, forecast_horizon)
        else:
            if self.return_proba:
                predicted = self.model_obj.predict_proba(X)
                predicted = predicted[:, 1]
            else:
                predicted = self.model_obj.predict(X)
        predicted = predicted.reshape(-1,1)
        if self.scale == True:
            predicted = self.scaler.inverse_transform(predicted)
        if self.linear_trend:
            linear_trend = [i for i in range(0, forecast_horizon)]
            linear_trend = np.reshape(linear_trend, (len(linear_trend), 1))
            linear_trend += len(self.scaled_y) + 1
            linear_trend = linear_trend**2
            linear_trend = np.multiply(linear_trend, self.slope*self.penalty) + self.intercept
            predicted = np.add(predicted, linear_trend.reshape(-1,1))
        return predicted

    def init_opt_params(self):
        if self.objective == 'regression':
            metric = 'rmse'
        elif self.objective == 'classification':
            metric = 'cross-entropy'
        self.opt_params = {
                                "objective": self.objective,
                                "metric": metric,
                                "verbosity": -1,
                                "boosting_type": "gbdt",
                                "seed": 42,
                                'linear_tree': False,
                            }

    def tree_optimize(self, y, exogenous=None, cv_splits=3, test_size=None):
        self.init_opt_params()
        if self.n_basis is not None:
            if len(y) <= self.n_basis - 1:
                self.n_basis = len(y) - 1
        self.exogenous = exogenous
        y = np.array(y)
        self.og_y = y
        if self.linear_trend:
            fitted_trend = self.linear_test(y)
            y = np.subtract(y, fitted_trend)
        # if self.linear_trend:
        #     self.ar = None
        #     self.decay = None
        if self.scale:
            self.scale_input(y)
        else:
            self.scaled_y = y.copy()
        self.X = self.build_input(self.scaled_y)
        study_tuner = optuna.create_study(direction='minimize')
        dtrain = lgb.Dataset(self.X, label=self.scaled_y)
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        tscv = TimeSeriesSplit(n_splits=cv_splits, test_size=test_size)
        tuner = lgb.LightGBMTunerCV(self.opt_params,
                                    dtrain,
                                    study=study_tuner,
                                    verbose_eval=False,
                                    early_stopping_rounds=10,
                                    seed = 42,
                                    folds=tscv,
                                    num_boost_round=500,
                                    show_progress_bar=False
                                    )
        
        tuner.run()
        best_params = tuner.best_params
        self.model_obj = gbm.LGBMRegressor(**best_params)
        self.model_obj.fit(self.X, self.scaled_y)
        fitted = self.model_obj.predict(self.X).reshape(-1,1)
        if self.scale:
            fitted = self.scaler.inverse_transform(fitted)
        if self.linear_trend:
            fitted = np.add(fitted.reshape(-1,1), fitted_trend.reshape(-1,1))
        return fitted

