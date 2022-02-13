# -*- coding: utf-8 -*-
import numpy as np


class FourierBasisFunction:

    def __init__(self, fourier_order):
        self.fourier_order = fourier_order

    def get_fourier_series(self, y, seasonal_period):
        x = 2 * np.pi * np.arange(1, self.fourier_order + 1) / seasonal_period
        t = np.arange(1, len(y) + 1)
        x = x * t[:, None]
        fourier_series = np.concatenate((np.cos(x), np.sin(x)), axis=1)
        return fourier_series
        # return np.column_stack([
        #             fun((2.0 * (i + 1) * np.pi * t / seasonal_period))
        #             for i in range(self.fourier_order)
        #             for fun in (np.sin, np.cos)
        #         ])

    
    def get_harmonics(self, y, seasonal_period):
        harmonics = self.get_fourier_series(y, seasonal_period)
        return harmonics

    def get_future_harmonics(self, harmonics, forecast_horizon, seasonal_period):
        total_length = len(harmonics) + forecast_horizon
        future_harmonics = self.get_fourier_series(np.arange(total_length), seasonal_period)
        return future_harmonics[len(harmonics):, :]

#%%
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('darkgrid')
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
    fbf = FourierBasisFunction(12, 5)
    f = fbf.get_harmonics(y)

