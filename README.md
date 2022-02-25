# LazyProphet
LazyProphet is a time series forecasting model built for LightGBM forecasting of single time series.

Many nice-ities have been added such as recursive forecasting when using lagged target variable such as the last 4 values to predict the 5th.

Additionally, fourier basis functions and penalized weighted piecewise linear basis functions are options as well!

Don't ever use in-sample fit for these types of models as they fit the data quite snuggly.

## Quickstart

```
pip install LazyProphet
```

Simple example from Sklearn:

```
from LazyProphet import LazyProphet as lp
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

bike_sharing = fetch_openml("Bike_Sharing_Demand", version=2, as_frame=True)
y = bike_sharing.frame['count']
y = y[-400:].values

lp_model = lp.LazyProphet(seasonal_period=[24, 168], #list means we use both seasonal periods
                          n_basis=4,
                          fourier_order=10,
                          ar=list(range(1,25)),
                          decay=.99
                          )
fitted = lp_model.fit(y)
predicted = lp_model.predict(100)

plt.plot(y)
plt.plot(np.append(fitted, predicted))
plt.axvline(400)
plt.show()
```
![alt text](https://github.com/tblume1992/LazyProphet/blob/main/static/example_output.png?raw=true "Output 1")
