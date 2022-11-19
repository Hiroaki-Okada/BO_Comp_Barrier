import pdb

import numpy as np

from gpytorch.priors import GammaPrior

from bo.bo_utils import BOUtils
from bo.surrogate_models import GPModel


class SecondBO(BOUtils):
    def __init__(self, obj, kernel_utils, real_row_results,
                 surrogate_model=GPModel, ard=False, acq=None,
                 deviation=None,
                 batch_size=5,
                 noise_prior=GammaPrior(1.5, 0.5),
                 noise_constraint=1e-5):

        self.all_name_combs = obj.all_name_combs
        self.target = obj.target
        self.target_scaling = obj.target_scaling
        self.opt_type = obj.opt_type
        self.gpu = obj.gpu

        self.kernel_utils = kernel_utils

        self.real_row_results = real_row_results

        self.obj = obj
        self.surrogate_model = surrogate_model
        self.ard = ard
        self.acq = acq
        self.deviation = deviation
        self.batch_size = batch_size
        self.noise_constraint = noise_constraint
        self.noise_prior = noise_prior

    def run(self):
        self.set_obj()

        best_kernel = self.get_kernel()
        proposed_idx = self.run_bo(kernel=best_kernel)

        return proposed_idx

    # Propose experimental batch
    def run_bo(self, kernel, n_restarts=0):
        surrogate_model = self.surrogate_model(self.obj.X,
                                               self.obj.y,
                                               kernel=kernel,
                                               gpu=self.gpu,
                                               n_restarts=n_restarts,
                                               noise_prior=self.noise_prior,
                                               noise_constraint=self.noise_constraint)

        surrogate_model.fit()

        idx_1 = self.obj.row_results.index.values
        idx_2 = self.real_row_results.index.values
        unique_idx = list(np.isin(idx_1, idx_2, invert=True))
        calc_only_idx = self.obj.row_results[unique_idx].index.values

        self.acq.function.batch_size = self.batch_size

        proposed_idx = self.acq.evaluate(surrogate_model,
                                         self.obj,
                                         kernel,
                                         calc_only_idx=calc_only_idx)

        return proposed_idx
