# LazyProphet v0.3.8

![alt text](https://github.com/tblume1992/LazyProphet/blob/main/LazyProphet/static/lp_logo.png "logo")

## Recent Changes

With v0.3.8 comes a fully fledged Optuna Optimizer for simple (no exogenous) regression problems. Classification is ToDo.

A Quick example of the new functionality:

```
from LazyProphet import LazyProphet as lp
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
y = bike_sharing.frame['count']
y = y[-400:].values

lp_model = lp.LazyProphet.Optimize(y,
                                seasonal_period=[24, 168],
                                n_folds=2, # must be greater than 1
                                n_trials=20, # number of optimization runs, default is 100
                                test_size=48 # size of the holdout set to test against
                                )
fitted = lp_model.fit(y)
predicted = lp_model.predict(100)

plt.plot(y)
plt.plot(np.append(fitted, predicted))
plt.axvline(400)
plt.show()
```

## Introduction

[A decent intro can be found here.](https://medium.com/p/3745bafe5ce5)

LazyProphet is a time series forecasting model built for LightGBM forecasting of single time series.

Many nice-ities have been added such as recursive forecasting when using lagged target variable such as the last 4 values to predict the 5th.

Additionally, fourier basis functions and penalized weighted piecewise linear basis functions are options as well!

Don't ever use in-sample fit for these types of models as they fit the data quite snuggly.

## Quickstart

```
pip install LazyProphet
```

Simple example from Sklearn, just give it the hyperparameters and an array:

```
from LazyProphet import LazyProphet as lp
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
y = bike_sharing.frame['count']
y = y[-400:].values

lp_model = lp.LazyProphet(seasonal_period=[24, 168], #list means we use both seasonal periods
                          n_basis=4, #weighted piecewise basis functions
                          fourier_order=10,
                          ar=list(range(1,25)),
                          decay=.99 #the 'penalized' in penalized weighted piecewise linear basis functions
                          )
fitted = lp_model.fit(y)
predicted = lp_model.predict(100)

plt.plot(y)
plt.plot(np.append(fitted, predicted))
plt.axvline(400)
plt.show()
```
![alt text](https://github.com/tblume1992/LazyProphet/blob/main/LazyProphet/static/example_output.png "Output 1")

If you are working with less data or then you will probably want to pass custom LightGBM params via boosting_params when creating the LazyProphet obj.

The default params are:

```
boosting_params = {
                        "objective": "regression",
                        "metric": "rmse",
                        "verbosity": -1,
                        "boosting_type": "gbdt",
                        "seed": 42,
                        'linear_tree': False,
                        'learning_rate': .15,
                        'min_child_samples': 5,
                        'num_leaves': 31,
                        'num_iterations': 50
                    }
```
*WARNING* 
Passing linear_tree=True can be extremely unstable, especially with ar and n_basis arguments. We do tests for linearity and will de-trend if necessary.
**

Most importantly for controlling the complexity by using num_leaves/learning_rate for complexity with less data.

Alternatively, you could try out the method:

```
tree_optimize(y, exogenous=None, cv_splits=3, test_size=None)
```
In-place of the fit method.  This will do 'cv_splits' number of Time-Series Cross-Validation steps to optimize the tree using Optuna. This method has some degraded performance in testing but may be better for autoforecasting various types of data sizes.
