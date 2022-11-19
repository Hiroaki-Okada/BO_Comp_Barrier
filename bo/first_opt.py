import pdb

from gpytorch.priors import GammaPrior

from bo.bo_utils import BOUtils
from bo.surrogate_models import GPModel


class FirstBO(BOUtils):
    def __init__(self, obj, kernel_utils,
                 surrogate_model=GPModel, ard=False, acq=None,
                 batch_size=5,
                 batch_magnification=1,
                 noise_prior=GammaPrior(1.5, 0.5),
                 noise_constraint=1e-5):

        self.all_name_combs = obj.all_name_combs
        self.target = obj.target
        self.target_scaling = obj.target_scaling
        self.opt_type = obj.opt_type
        self.gpu = obj.gpu

        self.kernel_utils = kernel_utils

        self.obj = obj
        self.surrogate_model = surrogate_model
        self.ard = ard
        self.acq = acq
        self.batch_size = batch_size
        self.batch_magnification = batch_magnification
        self.noise_constraint = noise_constraint
        self.noise_prior = noise_prior

    def run(self):
        self.set_obj()

        best_kernel = self.get_kernel()
        proposed_idx = self.run_bo(kernel=best_kernel)

        return proposed_idx

    # Propose computational batch
    def run_bo(self, kernel, n_restarts=0):
        surrogate_model = self.surrogate_model(self.obj.X,
                                               self.obj.y,
                                               kernel=kernel,
                                               gpu=self.gpu,
                                               n_restarts=n_restarts,
                                               noise_prior=self.noise_prior,
                                               noise_constraint=self.noise_constraint)

        surrogate_model.fit()

        self.acq.function.batch_size = self.batch_size * self.batch_magnification

        proposed_idx = self.acq.evaluate(surrogate_model, self.obj, kernel)

        return proposed_idx
