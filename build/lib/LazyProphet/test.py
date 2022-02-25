# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV, Lasso, Ridge
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
import optuna.integration.lightgbm as lgb
import optuna
from scipy import stats
import lightgbm as gbm
from sklearn.linear_model import LinearRegression
from LazyProphet.LinearBasisFunction import LinearBasisFunction
from LazyProphet.FourierBasisFunction import FourierBasisFunction

sns.set_style('darkgrid')

class LazyProphet:
    
    def __init__(self,
                 seasonal_period=None,
                 fourier_order=10,
                 n_changepoints=25,
                 ar=None,
                 decay=None,
                 scale=True,
                 weighted=True,
                 decay_average=False,
                 linear_trend=None,
                 boosting_params=None):
        self.exogenous = None
        if seasonal_period is not None:
            if not isinstance(seasonal_period, list):
                seasonal_period = [seasonal_period]
        self.seasonal_period = seasonal_period
        self.scale = scale
        if ar is not None:
            if not isinstance(ar, list):
                ar = [ar]
        self.ar = ar
        self.fourier_order = fourier_order
        self.decay = decay
        self.n_changepoints = n_changepoints
        self.weighted = weighted
        self.component_dict = {}
        self.decay_average = decay_average
        self.linear_trend = linear_trend
        if boosting_params is None:
            self.boosting_params = {
                                    "objective": "regression",
                                    "metric": "rmse",
                                    "verbosity": -1,
                                    "boosting_type": "gbdt",
                                    "seed": 42,
                                    'linear_tree': False,
                                    'learning_rate': .1,
                                    'min_child_samples': 5,
                                    'num_leaves': 31,

                                }
        else:
            self.boosting_params = boosting_params

    # def linear_test(self, y):
    #     y = y.copy().reshape((-1,))
    #     xi = np.arange(1, len(y) + 1)
    #     slope, intercept, r_value, p_value, std_err = stats.linregress(xi,y)
    #     trend_line = slope*xi*r_value + intercept
    #     if self.linear_trend is None:
    #         splitted_array = np.array_split(y.reshape(-1,), 4)
    #         mean_splits = np.array([np.mean(i) for i in splitted_array])
    #         asc_array = np.sort(mean_splits)
    #         desc_array = np.flip(asc_array)
    #         if all(asc_array == mean_splits):
    #             growth = True
    #         elif all(desc_array == mean_splits):
    #             growth = True
    #         else:
    #             growth = False
    #         if (r_value > .85 and growth):
    #             self.linear_trend = True
    #         else:
    #             self.linear_trend = False
    #     self.slope = slope * r_value
    #     self.intercept = intercept
    #     return trend_line

    def get_piecewise(self, y):
        self.lbf = LinearBasisFunction(n_changepoints=self.n_changepoints,
                                  decay=self.decay,
                                  weighted=self.weighted)
        basis = self.lbf.get_basis(y)
        return basis

    def get_harmonics(self, y, seasonal_period):
        self.fbf = FourierBasisFunction(self.fourier_order)
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

    def build_input(self, y, exogenous=None):
        self.basis = self.get_piecewise(y)
        X = self.basis
        self.component_dict['trend'] = self.basis
        if self.seasonal_period:
            for period in self.seasonal_period:
                harmonics = self.get_harmonics(y, period)
                self.component_dict['harmonics ' + str(period)] = harmonics
                X = np.append(X, harmonics, axis=1)
        if self.exogenous is not None:
            pass
        if self.ar is not None:
            for ar_order in self.ar:
                shifted_y = self.scaled_y.copy()
                shifted_y = LazyProphet.shift(shifted_y, ar_order)
                X = np.append(X, shifted_y.reshape(-1, 1), axis=1)
        return X

    def scale_input(self, y):
        self.scaler = StandardScaler()
        self.scaler.fit(np.asarray(y).reshape(-1, 1))
        self.scaled_y = y.copy()
        self.scaled_y = self.scaler.transform(self.scaled_y.reshape(-1, 1))

    def fit(self, y, exogenous=None):
        if len(y) >= self.n_changepoints - 1:
            self.changepoints = len(y) - 1
        self.exogenous = exogenous
        y = np.array(y)
        # if self.linear:
        #     self.ar = None
        #     self.decay = None
        self.og_y = y
        # if self.linear_trend:
        #     fitted_trend = self.linear_test(y)
        #     y = np.subtract(y, fitted_trend)
        if self.scale:
            self.scale_input(y)
        else:
            self.scaled_y = y.copy()
        self.X = self.build_input(self.scaled_y)
        self.model_obj = gbm.LGBMRegressor(**self.boosting_params)
        self.model_obj.fit(self.X, self.scaled_y.reshape(-1, ))
        fitted = self.model_obj.predict(self.X).reshape(-1,1)
        if self.scale:
            fitted = self.scaler.inverse_transform(fitted)
        # if self.linear_trend:
        #     fitted = np.add(fitted.reshape(-1,1), fitted_trend.reshape(-1,1))
        return fitted

    def recursive_predict(self, X, forecast_horizon):
        self.future_X = np.append(X, np.zeros((len(X), len(self.ar))), axis=1)
        predictions = []
        for step in range(forecast_horizon):
            for i, ar_order in enumerate(self.ar):
                column_slice = -len(self.ar) + i
                if step < ar_order:
                    self.future_X[step, column_slice] = self.scaled_y[-ar_order + step]
                else:
                    self.future_X[step, column_slice] = predictions[-ar_order]
            recursive_X = self.future_X[step, :].reshape(1, -1)
            predictions.append(self.model_obj.predict(recursive_X))
        return np.array(predictions)

    def predict(self, forecast_horizon, future_X=None):
        X = self.lbf.get_future_basis(self.component_dict['trend'],
                                      forecast_horizon,
                                      average=self.decay_average)
        if self.seasonal_period:
            for period in self.seasonal_period:
                harmonics = self.component_dict['harmonics ' + str(period)]
                future_harmonics = self.fbf.get_future_harmonics(harmonics,
                                                                 forecast_horizon,
                                                                 period)
                X = np.append(X, future_harmonics, axis=1)
        if self.exogenous is not None:
            pass
        if self.ar is not None:
            predicted = self.recursive_predict(X, forecast_horizon)
        else:
            predicted = self.model_obj.predict(X)
        predicted = predicted.reshape(-1,1)
        if self.scale == True:
            predicted = self.scaler.inverse_transform(predicted)
        # if self.linear_trend:
        #     linear_trend = [i for i in range(0, forecast_horizon)]
        #     linear_trend = np.reshape(linear_trend, (len(linear_trend), 1))
        #     linear_trend += len(self.scaled_y) + 1
        #     linear_trend = np.multiply(linear_trend, self.slope) + self.intercept
        #     predicted = np.add(predicted, linear_trend.reshape(-1,1))
        return predicted

    def optimize(self, y, cv_splits):
        study_tuner = optuna.create_study(direction='minimize')
        dtrain = lgb.Dataset(self.X, label=self.scaled_y)
        
        # Suppress information only outputs - otherwise optuna is 
        # quite verbose, which can be nice, but takes up a lot of space
        optuna.logging.set_verbosity(optuna.logging.WARNING) 
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        # Run optuna LightGBMTunerCV tuning of LightGBM with cross-validation
        tuner = lgb.LightGBMTunerCV(self.boosting_params,
                                    dtrain,
                                    study=study_tuner,
                                    verbose_eval=False,
                                    early_stopping_rounds=250,
                                    time_budget=19800, # Time budget of 5 hours, we will not really need it
                                    seed = 42,
                                    folds=tscv,
                                    num_boost_round=100,
                                    )
        
        tuner.run()
        best_params = tuner.best_params
        self.model_obj = gbm.LGBMRegressor(**best_params)
        self.model_obj.fit(self.X, self.scaled_y)
        fitted = self.model_obj.predict(self.X).reshape(-1,1)
        return fitted