# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 08:19:32 2022

@author: ER90614
"""
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
import lightgbm as gbm
from LazyProphet.LinearBasisFunction import LinearBasisFunction
from LazyProphet.FourierBasisFunction import FourierBasisFunction

sns.set_style('darkgrid')

class LazyProphet:
    
    def __init__(self,
                 trend_lambda=.005,
                 exogenous_lambda=.001,
                 seasonal_lambda=.001,
                 seasonal_period=None,
                 fourier_order=10,
                 n_changepoints=25,
                 decay=None,
                 scale=True,
                 weighted=True,
                 cv_splits=5,
                 decay_average=False,
                 boosting_params=None):
        self.exogenous = None
        if seasonal_period is not None:
            if not isinstance(seasonal_period, list):
                seasonal_period = [seasonal_period]
        self.seasonal_period = seasonal_period
        self.seasonal_lambda = seasonal_lambda
        self.exogenous_lambda = exogenous_lambda
        self.trend_lambda = trend_lambda
        self.scale = scale
        self.fourier_order = fourier_order
        self.decay = decay
        self.n_changepoints = n_changepoints
        self.weighted = weighted
        self.cv_splits = cv_splits
        self.component_dict = {}
        self.decay_average = decay_average
        if boosting_params is None:
            self.boosting_params = {
                                    "objective": "regression",
                                    "metric": "tweedie",
                                    "verbosity": -1,
                                    "boosting_type": "gbdt",
                                    "seed": 42,
                                    'linear_tree': False,
                                    'learning_rate': .1,
                                    'min_child_samples': 5,
                                    'num_leaves': 31
                                }
        else:
            self.boosting_params = boosting_params

    def get_trend(self, y):
        self.lbf = LinearBasisFunction(n_changepoints=self.n_changepoints,
                                  decay=self.decay,
                                  weighted=self.weighted)
        basis = self.lbf.get_basis(y)
        return basis

    def get_harmonics(self, y, seasonal_period):
        self.fbf = FourierBasisFunction(self.fourier_order)
        basis = self.fbf.get_harmonics(y, seasonal_period)
        return basis

    def build_input(self, y, exogenous=None):
        basis = self.get_trend(y)
        X = basis
        self.component_dict['trend'] = basis
        self.regularization_array = np.ones(np.shape(X)[1])*self.trend_lambda
        if self.seasonal_period:
            for period in self.seasonal_period:
                harmonics = self.get_harmonics(y, period)
                self.regularization_array = np.append(self.regularization_array,
                                                      np.ones(np.shape(harmonics)[1])*self.seasonal_lambda)
                self.component_dict['harmonics ' + str(period)] = harmonics
                X = np.append(X, harmonics, axis=1)
        if self.exogenous is not None:
            pass
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
        self.og_y = y
        if self.scale == True:
            self.scale_input(y)
        else:
            self.scaled_y = y.copy()
        self.X = self.build_input(self.scaled_y)
        # from sklearn.model_selection import TimeSeriesSplit
        # tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        
        # study_tuner = optuna.create_study(direction='minimize')
        # dtrain = lgb.Dataset(self.X, label=self.scaled_y)
        
        # # Suppress information only outputs - otherwise optuna is 
        # # quite verbose, which can be nice, but takes up a lot of space
        # optuna.logging.set_verbosity(optuna.logging.WARNING) 
        
        # # Run optuna LightGBMTunerCV tuning of LightGBM with cross-validation
        # tuner = lgb.LightGBMTunerCV(params, 
        #                             dtrain,
        #                             study=study_tuner,
        #                             verbose_eval=False,
        #                             early_stopping_rounds=250,
        #                             time_budget=19800, # Time budget of 5 hours, we will not really need it
        #                             seed = 42,
        #                             folds=tscv,
        #                             num_boost_round=1000,
        #                             )
        
        # tuner.run()
        # best_params = tuner.best_params
        # self.model_obj = gbm.LGBMRegressor(**best_params)
        # self.model_obj.fit(self.X, self.scaled_y)

        self.model_obj = gbm.LGBMRegressor(**self.boosting_params)
        self.model_obj.fit(self.X, self.scaled_y)
        fitted = self.model_obj.predict(self.X)
        if self.scale == True:
            fitted = self.scaler.inverse_transform(fitted.reshape(-1,1))
        # self.model_obj = LassoCV(cv=tscv,
        #                           fit_intercept=True,
        #                           # max_iter=1000000
        #                           )
        # self.model_obj = Ridge(
        #                           fit_intercept=True,
        #                           # max_iter=1000000
        #                           )
        # self.model_obj = sm.GLM(exog = self.X, endog = self.scaled_y)
        # self.model_obj = self.model_obj.fit_regularized(alpha = self.regularization_array)
        # self.model_obj = self.model_obj.fit()
        # self.model_obj.fit(X=self.X, y=self.scaled_y.ravel())
        # output = self.fit_predict(self.X)
        # if self.scale == True:
        #     fitted = self.scaler.inverse_transform(self.model_obj.fittedvalues.reshape(-1,1))
        return fitted
 
    def fit_predict(self, prediction_X):
        predicted = self.model_obj.predict(prediction_X)
        if self.scale == True:
            predicted = self.scaler.inverse_transform(predicted.reshape(-1,1))
        return predicted

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
        predicted = self.model_obj.predict(X)
        if self.scale == True:
            predicted = self.scaler.inverse_transform(predicted.reshape(-1,1))
        return predicted

    def plot_components(self):
        fig, ax = plt.subplots(len(self.component_dict.keys())+1)
        fig.tight_layout()
        for i, component in enumerate(list(self.component_dict.keys())):
            Test = self.X.copy()
            mask = np.repeat(True, np.shape(self.X)[1])
            mask[self.component_dict[component][0]:self.component_dict[component][1] + self.component_dict[component][0]] = False
            Test[:, mask] = 0
            component_prediction = self.fit_predict(Test)
            ax[i].plot(component_prediction)
            ax[i].set_title(component)
        residuals = self.og_y.reshape(-1,)-self.fit_predict(self.X).reshape(-1, )
        ax[i+1].scatter(np.array(np.arange(0, len(self.scaled_y))), residuals)
        ax[i+1].set_title('Residuals')
        plt.show()

    def plot_future_components(self):
        fig, ax = plt.subplots(len(self.component_dict.keys())+1)
        fig.tight_layout()
        for i, component in enumerate(list(self.component_dict.keys())):
            Test = self.prediction_X.copy()
            mask = np.repeat(True, np.shape(self.prediction_X)[1])
            mask[self.component_dict[component][0]:self.component_dict[component][1] + self.component_dict[component][0]] = False
            Test[:, mask] = 0
            component_prediction = self.predict(Test)
            ax[i].plot(component_prediction)
            ax[i].set_title(component)
        plt.show()

    def get_components(self):
        components = {}
        for i, component in enumerate(list(self.component_dict.keys())):
            Test = self.X.copy()
            mask = np.repeat(True, np.shape(self.X)[1])
            mask[self.component_dict[component][0]:self.component_dict[component][1] + self.component_dict[component][0]] = False
            Test[:, mask] = 0
            component_prediction = self.fit_predict(Test)
            components[component] = component_prediction
        return components

    def get_future_components(self):
        components = {}
        for i, component in enumerate(list(self.component_dict.keys())):
            Test = self.prediction_X.copy()
            mask = np.repeat(True, np.shape(self.prediction_X)[1])
            mask[self.component_dict[component][0]:self.component_dict[component][1] + self.component_dict[component][0]] = False
            Test[:, mask] = 0
            component_prediction = self.fit_predict(Test)
            components[component] = component_prediction
        
        return components
    
    def summary(self):
        return self.model_obj.summary()


