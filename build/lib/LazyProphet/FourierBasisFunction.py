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
