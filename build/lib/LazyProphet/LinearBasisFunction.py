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
