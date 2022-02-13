# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 08:19:32 2022

@author: ER90614
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

class LazyProphet:
    
    def __init__(self,y, 
                 exogenous = None, 
                 trend = True, 
                 changepoint = True, 
                 poly = 1,
                 scale_y = True,
                 scale_X = False,
                 trend_lambda = .01,
                 changepoint_lambda = .005,
                 exogenous_lambda = .001,
                 seasonal_lambda = .001,
                 linear_changepoint_lambda = .001,
                 seasonal_period = 12,
                 components = 10,
                 approximate_changepoints = True,
                 linear_changepoint = True,
                 n_changepoints = 25,
                 intercept = True,
                 pre = 1,
                 post = 1,
                 changepoint_depth = 2):
        self.y = y
        self.exogenous = exogenous
        self.trend = trend
        self.changepoint = changepoint
        self.poly = poly
        self.seasonal_period = seasonal_period
        self.seasonal_lambda = seasonal_lambda
        self.changepoint_lambda = changepoint_lambda
        self.exogenous_lambda = exogenous_lambda
        self.trend_lambda = trend_lambda
        self.scale_y = scale_y
        self.scale_X = scale_X
        self.components = components
        self.approximate_changepoints = approximate_changepoints
        self.linear_changepoint = linear_changepoint
        self.linear_changepoint_lambda = linear_changepoint_lambda
        self.pre = pre
        self.post = post
        self.intercept = intercept
        self.n_changepoints = n_changepoints
        self.changepoint_depth = changepoint_depth        
        
        return
        
    def get_fourier_series(self, t, p=12, n=10):
        x = 2 * np.pi * np.arange(1, n + 1) / p
        x = x * t[:, None]
        fourier_series = np.concatenate((np.cos(x), np.sin(x)), axis=1)
        
        return fourier_series
    
    def get_harmonics(self):
        harmonics = self.get_fourier_series(np.arange(len(self.y)), 
                                self.seasonal_period, 
                                n = self.components)        
        return harmonics

    def get_future_harmonics(self):
        harmonics = self.get_fourier_series(np.arange(len(self.y) + self.n_steps), 
                                self.seasonal_period, 
                                n = self.components)
        
        return harmonics
    
    def get_changepoint(self):
        changepoints = np.zeros(shape=(len(self.y),int(len(self.y))))
        for i in range(int(len(self.y) - 1)):
            changepoints[:i + 1, i] = 1        
        return pd.DataFrame(changepoints)
    
    def get_linear_changepoint(self, future_steps = 0):
        array_splits = np.array_split(np.array(self.y),self.n_changepoints)
        y = np.array(self.y)
        initial_point = y[0]
        final_point = y[-1]
        changepoints = np.zeros(shape=(len(y) + future_steps,self.n_changepoints))
        for i in range(self.n_changepoints):
            moving_point = array_splits[i][-1]
            if i == 0:
                len_splits = len(array_splits[i])
            else:
                len_splits = len(np.concatenate(array_splits[:i+1]).ravel())
            slope = (moving_point - initial_point)/(len_splits)
            slope = slope*self.pre
            if i != self.n_changepoints - 1:        
                reverse_slope = (final_point - moving_point)/(len(y) - len_splits)
                reverse_slope = reverse_slope*self.post
            
            changepoints[0:len_splits, i] = slope * (1+np.array(list(range(len_splits))))
            changepoints[len_splits:, i] = changepoints[len_splits-1, i] + reverse_slope * (1+np.array(list(range((len(y) + future_steps - len_splits)))))
        if future_steps:
            changepoints[len(self.y):, :] = np.array([np.mean(changepoints[len(self.y):, :], axis = 1)]*np.shape(changepoints)[1]).transpose()
        return changepoints 
        
    
    def get_approximate_changepoints(self):
        from sklearn import tree
        changepoints = np.zeros(shape=(len(self.y),int(len(self.y))))
        for i in range(int(len(self.y) - 1)):
            changepoints[:i + 1, i] = 1
        changepoints = pd.DataFrame(changepoints)
        clf = tree.DecisionTreeRegressor(criterion = 'mae', max_depth = 2)
        clf = clf.fit(changepoints, self.y)
        imp = pd.Series(clf.feature_importances_)
        imp_idx = imp[imp > .01]
        model_dataset = changepoints.iloc[:, imp_idx.index]
        return model_dataset
    
    def get_trend(self):
        n = len(self.y)
        linear_trend = np.arange(len(self.y))
        trends = np.asarray(linear_trend).reshape(n, 1)
        trends = np.append(trends, np.asarray(linear_trend**self.poly).reshape(n, 1), axis = 1)
        return trends

    def get_future_trend(self):
        n = len(self.y) + self.n_steps
        linear_trend = np.arange(n)
        trends = np.asarray(linear_trend).reshape(n, 1)
        trends = np.append(trends, np.asarray(linear_trend**self.poly).reshape(n, 1), axis = 1)

        return trends
    
    def fit(self):
        if self.seasonal_period:
            X = self.get_harmonics()
            regularization_array = np.ones(np.shape(X)[1])*self.seasonal_lambda
            component_dict = {'harmonics': [0, np.shape(X)[1] - 1]}
        elif self.trend:
            X = self.get_trend()
            component_dict = {'trends': [0, np.shape(X)[1] - 1]}
            regularization_array = np.ones(np.shape(X)[1])*self.trend_lambda           
        if self.trend and self.seasonal_period:
            trends = self.get_trend()
            component_dict['trends'] = [np.shape(X)[1], np.shape(trends)[1]]
            X = np.append(X, trends, axis = 1)
            regularization_array = np.append(regularization_array, 
                                             np.ones(np.shape(trends)[1])*self.trend_lambda)
            
        if self.changepoint:
            if self.approximate_changepoints:
                changepoints = self.get_approximate_changepoints()
            else:   
                changepoints = self.get_changepoint()
            self.changepoints = changepoints
            try:
                component_dict['changepoint'] = [np.shape(X)[1], np.shape(changepoints)[1]]
                X = np.append(X, changepoints, axis = 1)
                regularization_array = np.append(regularization_array, 
                                             np.ones(np.shape(changepoints)[1])*self.changepoint_lambda)
            except:
                X = self.changepoints
                component_dict = {'changepoint': [0, np.shape(X)[1] - 1]}
                regularization_array = np.ones(np.shape(changepoints)[1])*self.changepoint_lambda
                X = np.array(X)

        
        if self.scale_X:
            self.X_scaler = StandardScaler()
            self.X_scaler = self.X_scaler.fit(X)
            X = self.X_scaler.transform(X)
        if self.scale_y:
            self.scaler = StandardScaler()
            self.scaler.fit(np.asarray(self.y).reshape(-1, 1))   
            self.og_y = self.y.copy()
            self.y = self.scaler.transform(np.asarray(self.y).reshape(-1, 1))

       
        if self.exogenous is not None:
            component_dict['exogenous'] = [np.shape(X)[1], np.shape(self.exogenous)[1]]
            X = np.append(X, self.exogenous, axis = 1)
            regularization_array = np.append(regularization_array, 
                                             np.ones(np.shape(self.exogenous)[1])*self.exogenous_lambda)
            
        if self.linear_changepoint:
            self.linear_changepoints = self.get_linear_changepoint()
            component_dict['linear_changepoint'] = [np.shape(X)[1], np.shape(self.linear_changepoints)[1]]
            X = np.append(X, self.linear_changepoints, axis = 1)
            regularization_array = np.append(regularization_array, 
                                             np.ones(np.shape(self.linear_changepoints)[1])*self.linear_changepoint_lambda)            
        if self.intercept:
            intercept_term = np.asarray(np.ones(len(self.y))).reshape(-1, 1)
            component_dict['intercept'] = [np.shape(X)[1], np.shape(intercept_term)[1]]
            regularization_array = np.append(regularization_array, 
                                             np.ones(np.shape(intercept_term)[1])*self.trend_lambda)
            X = np.append(X, intercept_term, axis = 1)
        lasso = ElasticNetCV(cv=5, fit_intercept=False, max_iter=1000000)
        lasso.fit(X=X, y=self.y.ravel())
        self.lasso = lasso
        self.component_dict = component_dict
        self.X = X
        
        return  
 
    def predict(self, prediction_X = None):
        if prediction_X is None:
            prediction_X = self.X    
        predicted = self.lasso.predict(prediction_X)
        if self.scale_y == True:
            predicted = self.scaler.inverse_transform(predicted.reshape(-1,))
            
        return predicted
    
    def make_future_dataframe(self, n_steps, exogenous = None):
        self.n_steps = n_steps
        if self.seasonal_period:
            X = self.get_future_harmonics()
        else:
            X = self.get_future_trend()
        X = X[-n_steps:, :]
        if self.trend and self.seasonal_period:
            trends = self.get_future_trend()
            trends = trends[-n_steps:, :]
            X = np.append(X, trends, axis = 1)
            
        if self.changepoint:
            future_changepoints = self.changepoints.iloc[-1:, :]
            future_changepoints = pd.concat([future_changepoints]*n_steps, ignore_index=True)
            X = np.append(X, future_changepoints, axis = 1)
 
        if self.scale_X:
            X = self.X_scaler.transform(X)            

        if self.exogenous is not None:
            X = np.append(X, exogenous, axis = 1)


        if self.linear_changepoint:
            future_linear_changepoints = self.get_linear_changepoint(n_steps)
            future_linear_changepoints = future_linear_changepoints[-n_steps:, :]
            X = np.append(X, future_linear_changepoints, axis = 1)
        if self.intercept:
            intercept_term = np.asarray(np.ones(n_steps)).reshape(n_steps, 1)
            X = np.append(X, intercept_term, axis = 1)
        self.prediction_X = X
            
        return X
            
    def plot_components(self):
        fig, ax = plt.subplots(len(self.component_dict.keys()), figsize = (16,16))
        for i, component in enumerate(list(self.component_dict.keys())):
            Test = self.X.copy()
            mask = np.repeat(True, np.shape(self.X)[1])
            mask[self.component_dict[component][0]:self.component_dict[component][1] + self.component_dict[component][0]] = False
            Test[:, mask] = 0
            component_prediction = self.predict(Test)
            if self.scale_y and component != 'intercept':
                component_prediction = component_prediction - np.mean(self.og_y)
            ax[i].plot(component_prediction)
            ax[i].set_title(component)
        plt.show()

    def plot_future_components(self):
        fig, ax = plt.subplots(len(self.component_dict.keys()), figsize = (16,16))
        for i, component in enumerate(list(self.component_dict.keys())):
            Test = self.prediction_X.copy()
            mask = np.repeat(True, np.shape(self.prediction_X)[1])
            mask[self.component_dict[component][0]:self.component_dict[component][1] + self.component_dict[component][0]] = False
            Test[:, mask] = 0
            component_prediction = self.predict(Test)
            if self.scale_y and component != 'intercept':
                component_prediction = component_prediction - np.mean(self.og_y)
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
            component_prediction = self.predict(Test)
            if self.scale_y and component != 'intercept':
                component_prediction = component_prediction - np.mean(self.og_y)
            components[component] = component_prediction
        
        return components

    def get_future_components(self):
        components = {}
        for i, component in enumerate(list(self.component_dict.keys())):
            Test = self.prediction_X.copy()
            mask = np.repeat(True, np.shape(self.prediction_X)[1])
            mask[self.component_dict[component][0]:self.component_dict[component][1] + self.component_dict[component][0]] = False
            Test[:, mask] = 0
            component_prediction = self.predict(Test)
            if self.scale_y and component != 'intercept':
                component_prediction = component_prediction - np.mean(self.og_y)
            components[component] = component_prediction
        
        return components
    
    def summary(self):
        return self.lasso.summary()

# %%
if __name__ == '__main__':

    def _into_subchunks(x, subchunk_length, every_n=1):
        """
        Split the time series x into subwindows of length "subchunk_length", starting every "every_n".
    
        For example, the input data if [0, 1, 2, 3, 4, 5, 6] will be turned into a matrix
    
            0  2  4
            1  3  5
            2  4  6
    
        with the settings subchunk_length = 3 and every_n = 2
        """
        len_x = len(x)
    
        assert subchunk_length > 1
        assert every_n > 0
    
        # how often can we shift a window of size subchunk_length over the input?
        num_shifts = (len_x - subchunk_length) // every_n + 1
        shift_starts = every_n * np.arange(num_shifts)
        indices = np.arange(subchunk_length)
    
        indexer = np.expand_dims(indices, axis=0) + np.expand_dims(shift_starts, axis=1)
        return np.asarray(x)[indexer]

    def sample_entropy(x):
        """
        Calculate and return sample entropy of x.
    
        .. rubric:: References
    
        |  [1] http://en.wikipedia.org/wiki/Sample_Entropy
        |  [2] https://www.ncbi.nlm.nih.gov/pubmed/10843903?dopt=Abstract
    
        :param x: the time series to calculate the feature of
        :type x: numpy.ndarray
    
        :return: the value of this feature
        :return type: float
        """
        x = np.array(x)
    
        # if one of the values is NaN, we can not compute anything meaningful
        if np.isnan(x).any():
            return np.nan
    
        m = 2  # common value for m, according to wikipedia...
        tolerance = 0.2 * np.std(
            x
        )  # 0.2 is a common value for r, according to wikipedia...
    
        # Split time series and save all templates of length m
        # Basically we turn [1, 2, 3, 4] into [1, 2], [2, 3], [3, 4]
        xm = _into_subchunks(x, m)
    
        # Now calculate the maximum distance between each of those pairs
        #   np.abs(xmi - xm).max(axis=1)
        # and check how many are below the tolerance.
        # For speed reasons, we are not doing this in a nested for loop,
        # but with numpy magic.
        # Example:
        # if x = [1, 2, 3]
        # then xm = [[1, 2], [2, 3]]
        # so we will substract xm from [1, 2] => [[0, 0], [-1, -1]]
        # and from [2, 3] => [[1, 1], [0, 0]]
        # taking the abs and max gives us:
        # [0, 1] and [1, 0]
        # as the diagonal elements are always 0, we substract 1.
        B = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= tolerance) - 1 for xmi in xm])
    
        # Similar for computing A
        xmp1 = _into_subchunks(x, m + 1)
    
        A = np.sum(
            [np.sum(np.abs(xmi - xmp1).max(axis=1) <= tolerance) - 1 for xmi in xmp1]
        )
    
        # Return SampEn
        return -np.log(A / B)

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
    skylasso_model = SkyLasso(y, 
                                    changepoint = True,
                                    trend = True,
                                    poly = 2,
                                    linear_changepoint = True,
                                    seasonal_period = 12,
                                    components = 4,
                                    changepoint_depth = 3,
                                    scale_y = True,
                                    scale_X = False,
                                    intercept = True,
                                    approximate_changepoints = False,
                                    n_changepoints = 25
                                    )
    
        
    skylasso_model.fit()
    predicted = skylasso_model.predict()
    future_df = skylasso_model.make_future_dataframe(24)
    forecast = skylasso_model.predict(future_df)  
    skylasso_model.plot_components()
    components = skylasso_model.X
    plt.plot(np.append(predicted, forecast))
    plt.plot(y.values)

# %%

    def get_linear_changepoint(y, n_changepoints, pre=1, post=1, future_steps = 100, prediction_average=False):
        if post < 1:
            prediction_average = False
        array_splits = np.array_split(np.array(y), n_changepoints)
        y = np.array(y)
        initial_point = y[0]
        final_point = y[-1]
        changepoints = np.zeros(shape=(len(y) + future_steps, n_changepoints))
        for i in range(n_changepoints):
            moving_point = array_splits[i][-1]
            if i == 0:
                len_splits = len(array_splits[i])
            else:
                len_splits = len(np.concatenate(array_splits[:i+1]).ravel())
            slope = (moving_point - initial_point)/(len_splits)
            slope = slope*pre
            if i != n_changepoints - 1:
                reverse_slope = (final_point - moving_point)/(len(y) - len_splits)
                reverse_slope = reverse_slope*post
            
            changepoints[0:len_splits, i] = slope * (1+np.array(list(range(len_splits))))
            changepoints[len_splits:, i] = changepoints[len_splits-1, i] + reverse_slope * (1+np.array(list(range((len(y) + future_steps - len_splits)))))
        if future_steps and prediction_average:
            changepoints[len(y):, :] = np.array([np.mean(changepoints[len(y):, :], axis = 1)]*np.shape(changepoints)[1]).transpose()
        return changepoints[:len(y), :], changepoints[len(y): ,:]

    cp, future_cp = get_linear_changepoint(y, n_changepoints=25, pre=1, post=1, prediction_average=True)
    plt.plot(np.append(cp, future_cp, axis=0) + y.values[0])
    plt.plot(y.values)

    from sklearn.linear_model import LassoCV, ElasticNetCV, RidgeCV
    import quandl
    data = quandl.get("BITSTAMP/USD")
    y = data['High']
    y = pd.Series(y[-450:].values)
    regr = ElasticNetCV(cv=5, fit_intercept=True, max_iter=1000000)
    avg_forecast = []
    for i in range(500):
        cp, future_cp = get_linear_changepoint(y, n_changepoints=115, pre=1, post=.1, prediction_average=False)
        regr.fit(cp, y.values)
        forecast = np.append(regr.predict(cp), regr.predict(future_cp))
        plt.plot(forecast, label=i)
        avg_forecast.append(forecast)
    plt.plot(y.values)
    mean_forecast = np.mean(avg_forecast, axis=0)
    std_forecast = np.std(avg_forecast, axis=0)
    plt.plot(mean_forecast)
    plt.fill_between(np.arange(550), mean_forecast + 3*std_forecast, y2=mean_forecast - 3*std_forecast, alpha=.2)
    plt.plot(y.values)
    coefficients = regr.coef_
    def MASE(training_series, prediction_series, testing_series):
        """
        Computes the MEAN-ABSOLUTE SCALED ERROR forcast error for univariate time series prediction.
        
        See "Another look at measures of forecast accuracy", Rob J Hyndman
        
        parameters:
            training_series: the series used to train the model, 1d numpy array
            testing_series: the test series to predict, 1d numpy array or float
            prediction_series: the prediction of testing_series, 1d numpy array (same size as testing_series) or float
            absolute: "squares" to use sum of squares and root the result, "absolute" to use absolute values.
        
        """
        n = training_series.shape[0]
        d = np.abs(  np.diff( training_series) ).sum()/(n-1)
        
        errors = np.abs(testing_series - prediction_series )
        return errors.mean()/d

    # try:
    # row = 4

    y = train_df.iloc[row, :].dropna()
    y = y.iloc[-(3*seasonality):]
    y_test = test_df.iloc[row, :].dropna()
    seasonal_periods = len(y) / seasonality
    seasonal_sample_weights = []
    weight = 1
    for i in range(len(y)):
        if (i) % seasonality == 0:
            weight += 1
        seasonal_sample_weights.append(weight)
    exo = np.resize(np.array(y.values), (len(y), 1))

def get_seasonal_weights(y, seasonality):
    seasonal_periods = len(y) / seasonality
    seasonal_sample_weights = []
    weight = 0
    for i in range(len(y)):
        if (i) % seasonality == 0:
            weight += 1
        seasonal_sample_weights.append(weight)
    return seasonal_sample_weights
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


def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
smapes = []
naive_smape = []
j = tqdm(range(len(train_df)))
seasonality = 52
trend_estimator=[['linear', 'ses'],
                 ['linear', 'damped_des'],
                 ['linear', 'des']]
for row in j:
    y = train_df.iloc[row, :].dropna()
    y = y.iloc[-(3*seasonality):]
    y_test = test_df.iloc[row, :].dropna()
    j.set_description(f'{np.mean(smapes)}, {np.mean(naive_smape)}')