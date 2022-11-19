import pdb

import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal


class ExactGPModel(ExactGP):
    def __init__(self, train_X, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_X, train_y, likelihood)

        self.mean_module = ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return MultivariateNormal(mean_x, covar_x)


def fast_computation(isFast):
    gpytorch.settings.fast_pred_var._state = isFast
    gpytorch.settings.fast_pred_samples._state = isFast
    gpytorch.settings.fast_computations.covar_root_decomposition._state = isFast
    gpytorch.settings.fast_computations.log_prob._state = isFast
    gpytorch.settings.fast_computations.solves._state = isFast
    gpytorch.settings.deterministic_probes._state = isFast
    gpytorch.settings.memory_efficient._state = isFast
