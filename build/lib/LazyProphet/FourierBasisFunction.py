# -*- coding: utf-8 -*-
import numpy as np


class FourierBasisFunction:

    def __init__(self, fourier_order, seasonal_weights=None):
        self.fourier_order = fourier_order
        self.seasonal_weights = seasonal_weights
        if self.seasonal_weights is not None:
            self.seasonal_weights = np.array(self.seasonal_weights).reshape((-1, 1))

    def get_fourier_series(self, y, seasonal_period):
        x = 2 * np.pi * np.arange(1, self.fourier_order + 1) / seasonal_period
        t = np.arange(1, len(y) + 1)
        x = x * t[:, None]
        fourier_series = np.concatenate((np.cos(x), np.sin(x)), axis=1)
        return fourier_series
    
    def get_harmonics(self, y, seasonal_period):
        harmonics = self.get_fourier_series(y, seasonal_period)
        if self.seasonal_weights is not None:
            harmonics = harmonics * self.seasonal_weights
        return harmonics

    def get_future_harmonics(self, harmonics, forecast_horizon, seasonal_period):
        total_length = len(harmonics) + forecast_horizon
        future_harmonics = self.get_fourier_series(np.arange(total_length), seasonal_period)
        if self.seasonal_weights is None:
            return future_harmonics[len(harmonics):, :]
        else:
            return future_harmonics[len(harmonics):, :] * self.seasonal_weights[-1]

#%%
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from ThymeBoost import ThymeBoost as tb
    
    sns.set_style('darkgrid')
    
    #Here we will just create a random series with seasonality and a slight trend
    seasonality = ((np.cos(np.arange(1, 101))*10 + 50))
    seasonality[50:] = seasonality[50:] * 2 - np.mean(seasonality)
    np.random.seed(100)
    noise = np.random.normal(0, 1, 100)
    y = noise + seasonality
    plt.plot(y)
    plt.show()
    fbf = FourierBasisFunction(1)
    basis = fbf.get_harmonics(y, 25)
    plt.plot(basis)

    basis = fbf.get_future_harmonics(basis, 100,25)
    plt.plot(basis)
