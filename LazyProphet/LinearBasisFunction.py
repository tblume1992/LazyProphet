# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


class LinearBasisFunction:

    def __init__(self, n_changepoints, decay=None, weighted=True):
        self.n_changepoints = n_changepoints
        self.decay = decay
        self.weighted = weighted

    def get_basis(self, y):
        y = y.copy()
        y -= y[0]
        n_changepoints = self.n_changepoints
        array_splits = np.array_split(np.array(y),n_changepoints + 1)[:-1]
        if self.weighted:
            initial_point = y[0]
            final_point = y[-1]
        else:
            initial_point = 0
            final_point = 0
        changepoints = np.zeros(shape=(len(y), n_changepoints))
        len_splits = 0
        for i in range(n_changepoints):
            len_splits += len(array_splits[i])
            if self.weighted:
                moving_point = array_splits[i][-1]
            else:
                moving_point = 1
            left_basis = np.linspace(initial_point,
                                     moving_point,
                                     len_splits)
            end_point = self.add_decay(moving_point, final_point)
            right_basis = np.linspace(moving_point,
                                      end_point,
                                      len(y) - len_splits + 1)
            changepoints[:, i] = np.append(left_basis, right_basis[1:])
        return changepoints

    def add_decay(self, moving_point, final_point):
            if self.decay is None:
                return final_point
            else:
                return moving_point - ((moving_point - final_point) * (1 - self.decay))

    def get_future_basis(self, basis_functions, forecast_horizon, average=False):
        n_components = np.shape(basis_functions)[1]
        slopes = np.gradient(basis_functions)[0][-1, :]
        future_basis = np.array(np.arange(0, forecast_horizon + 1))
        future_basis += len(basis_functions)
        future_basis = np.transpose([future_basis] * n_components)
        future_basis = future_basis * slopes
        future_basis = future_basis + (basis_functions[-1, :] - future_basis[0, :])
        if average:
            future_basis = np.transpose([np.mean(future_basis, axis=1)] * n_components)
        return future_basis[1:, :]
# %%
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
    lbf = LinearBasisFunction(n_changepoints=10, decay=None, weighted=False)
    changepoints = lbf.get_basis(y)
    plt.plot(changepoints)
    future_changepoints = lbf.get_future_basis(changepoints, 24, average=False)
    plt.plot(np.append(changepoints, future_changepoints, axis=0))
