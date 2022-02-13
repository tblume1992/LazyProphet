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

# %%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    training_y = [
        5541,
        2371,
        1778,
        2310,
        1961,
        3486,
        17558,
        10382,
        6881,
        934,
        969,
        4166,
        3802,
        2756,
        2516,
        2994,
        2298,
        8336,
        7558,
        17071,
        8024,
        1780,
        1506,
        4494,
        7541,
        1079,
        1071,
        2510,
        1525,
        4102,
        7375,
        10966,
        3126,
        2664,
        4623,
        237,
        ]
    training_months = [
        '2017-12-31T00:00:00.000Z',
        '2018-01-31T00:00:00.000Z',
        '2018-02-28T00:00:00.000Z',
        '2018-03-31T00:00:00.000Z',
        '2018-04-30T00:00:00.000Z',
        '2018-05-31T00:00:00.000Z',
        '2018-06-30T00:00:00.000Z',
        '2018-07-31T00:00:00.000Z',
        '2018-08-31T00:00:00.000Z',
        '2018-09-30T00:00:00.000Z',
        '2018-10-31T00:00:00.000Z',
        '2018-11-30T00:00:00.000Z',
        '2018-12-31T00:00:00.000Z',
        '2019-01-31T00:00:00.000Z',
        '2019-02-28T00:00:00.000Z',
        '2019-03-31T00:00:00.000Z',
        '2019-04-30T00:00:00.000Z',
        '2019-05-31T00:00:00.000Z',
        '2019-06-30T00:00:00.000Z',
        '2019-07-31T00:00:00.000Z',
        '2019-08-31T00:00:00.000Z',
        '2019-09-30T00:00:00.000Z',
        '2019-10-31T00:00:00.000Z',
        '2019-11-30T00:00:00.000Z',
        '2019-12-31T00:00:00.000Z',
        '2020-01-31T00:00:00.000Z',
        '2020-02-29T00:00:00.000Z',
        '2020-03-31T00:00:00.000Z',
        '2020-04-30T00:00:00.000Z',
        '2020-05-31T00:00:00.000Z',
        '2020-06-30T00:00:00.000Z',
        '2020-07-31T00:00:00.000Z',
        '2020-08-31T00:00:00.000Z',
        '2020-09-30T00:00:00.000Z',
        '2020-10-31T00:00:00.000Z',
        '2020-11-30T00:00:00.000Z',
        ]
    training_months = [pd.to_datetime(i) for i in training_months]
    y = pd.Series(training_y, index = training_months)

    from sklearn.datasets import fetch_openml
    
    bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
    df = bike_sharing.frame
    y = df['count'][-200:]
    lp_model = LazyProphet(scale=True,
                           n_changepoints=25,
                           seasonal_period=[24, 168],
                           fourier_order=10,
                           decay=None,
                           decay_average=True)
    fitted = lp_model.fit(y)
    plt.plot(y.values)
    predictions = lp_model.predict(300)
    plt.plot(np.append(fitted, predictions))


    from fbprophet import Prophet
    prophet_train_df = y.reset_index()
    prophet_train_df.columns = ['ds', 'y']
    prophet_train_df['ds'] = prophet_train_df['ds'].dt.tz_localize(None)
    prophet = Prophet(seasonality_mode='additive')
    prophet.fit(prophet_train_df)
    future_df = prophet.make_future_dataframe(periods=24, freq='M')
    prophet_forecast = prophet.predict(future_df)
    plt.plot(prophet_forecast['yhat'])
    plt.plot(np.append(fitted, predictions))
    plt.plot(y.values)
    prophet.plot_components(prophet_forecast)

    def get_fourier_series(y, seasonal_period):
        x = 2 * np.pi * np.arange(1, 3 + 1) / seasonal_period
        t = np.arange(1, 1+ len(y))
        x = x * t[:, None]
        fourier_series = np.concatenate((np.cos(x), np.sin(x)), axis=1)
        return fourier_series
    plt.plot(get_fourier_series(y, 12))
    y.index = y.index.tz_localize(None)
    
    plt.plot(fourier_series(y.index, 12, 3))
    from datetime import timedelta,datetime


# %%
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    import pandas as pd
    from ThymeBoost  import ThymeBoost as tb
    import pmdarima
    
    train_df = pd.read_csv(r'C:\Users\er90614\Downloads\m4-weekly-train.csv')
    test_df = pd.read_csv(r'C:\Users\er90614\Downloads\m4-weekly-test.csv')
    train_df.index = train_df['V1']
    train_df = train_df.drop('V1', axis = 1)
    test_df.index = test_df['V1']
    test_df = test_df.drop('V1', axis = 1)
    
    from datetime import datetime, timedelta
    
    date_today = datetime.now()
    days = pd.date_range(date_today, date_today + timedelta(10000), freq='W')
    def smape(A, F):
        return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
    smapes = []
    naive_smape = []
    prophet_smape = []
    j = tqdm(range(len(train_df)))
    seasonality = 52
    trend_estimator=[['linear', 'ses'],
                     ['linear', 'damped_des'],
                     ['linear', 'des']]
    for row in j:
        y = train_df.iloc[row, :].dropna()
        # y = y.iloc[-(3*seasonality):]
        y_test = test_df.iloc[row, :].dropna()
        j.set_description(f'{np.mean(smapes)}, {np.mean(prophet_smape)}, {np.mean(naive_smape)}')
        lp_model = LazyProphet(scale=True,
                               seasonal_period=52,
                               n_changepoints=10,
                               fourier_order=10,
                               decay=.9,
                               decay_average=False,
                               weighted=False)
        fitted = lp_model.fit(y)
        predictions = lp_model.predict(len(y_test))
        # y.index = days[:len(y)]
        # prophet_train_df = y.reset_index()
        # prophet_train_df.columns = ['ds', 'y']
        # # prophet_train_df['ds'] = prophet_train_df['ds'].dt.tz_localize(None)
        # prophet = Prophet(seasonality_mode='additive')
        # prophet.fit(prophet_train_df)
        # future_df = prophet.make_future_dataframe(periods=len(y_test), freq='M')
        # prophet_forecast = prophet.predict(future_df)
        # prophet_forecast = prophet_forecast['yhat'].values[-len(y_test):]
        # prophet_smape.append(smape(y_test.values, pd.Series(prophet_forecast.reshape(-1)).clip(lower=0)))
        smapes.append(smape(y_test.values, pd.Series(predictions.reshape(-1)).clip(lower=0)))
        naive_smape.append(smape(y_test.values, np.tile(y.iloc[-1], len(y_test))))  
print(np.mean(smapes))
print(np.mean(naive_smape))
        
#%%
    import yfinance as yf
    from fbprophet import Prophet
    data = yf.download(tickers='BTC-USD', period = '3200h', interval = '60m')
    y = data['High'].iloc[-500:]
    y.index = pd.to_datetime(y.index)
    y.index = y.index.tz_localize(None)
    plt.plot(y)

    lp_model = LazyProphet(scale=True,
                           seasonal_period=[168,24],
                           fourier_order=10,
                           decay=.9,
                           decay_average=False)
    fitted = lp_model.fit(y)
    predictions = lp_model.predict(300)
    components = lp_model.get_components()

    lp_model.plot_components()
    look = lp_model.X
    plt.plot(look)
    prophet_train_df = y.reset_index()
    prophet_train_df.columns = ['ds', 'y']
    # prophet_train_df['ds'] = prophet_train_df['ds'].dt.tz_localize(None)
    prophet = Prophet(seasonality_mode='additive')
    prophet.fit(prophet_train_df)
    future_df = prophet.make_future_dataframe(periods=300, freq='H')
    prophet_forecast = prophet.predict(future_df)
    prophet.plot_components(prophet_forecast)
    plt.plot(y.values)
    plt.plot(prophet_forecast['yhat'].values)
    plt.plot(np.append(fitted, predictions))
