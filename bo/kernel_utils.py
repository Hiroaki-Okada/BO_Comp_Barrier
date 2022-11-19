import pdb

from copy import deepcopy

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from gpytorch.priors import GammaPrior
from gpytorch.kernels import RBFKernel, MaternKernel, LinearKernel, PolynomialKernel, ScaleKernel
from gpytorch.constraints import GreaterThan

from bo.pd_utils import to_tensor
from bo.surrogate_models import GPModel


class KernelUtils:
    def __init__(self, kernel_opt=False, ard=False, num_dims=None, gpu=False,
                 lengthscale_prior=None, outputscale_prior=None,
                 variance_prior=None, offset_prior=None,
                 noise_prior=None, noise_constraint=1e-5):

        self.kernel_opt = kernel_opt
        self.ard = ard
        self.num_dims = num_dims
        self.gpu = gpu
        self.noise_constraint = noise_constraint
        self.noise_prior = noise_prior
        self.variance_prior = variance_prior
        self.offset_prior = offset_prior
        self.lengthscale_prior = lengthscale_prior
        self.outputscale_prior = outputscale_prior

    @property
    def best_kernel(self):
        return self._best_kernel

    def opt_kernel(self, X, y):
        self._check_status()

        self.X = deepcopy(X)
        self.y = deepcopy(y)

        if self.kernel_opt == True:
            best_kernel = self.get_best_kernel()

        elif self.kernel_opt == False:
            best_kernel = self.get_default_kernel()

        else:
            best_kernel = self.get_specific_kernel()

        self._best_kernel = best_kernel

    def _check_status(self):
        if self.ard and self.num_dims is None:
            raise ValueError('Invalid ARD kernel setting')

        if self.kernel_opt not in [True, False] and \
           self.kernel_opt.lower() not in ['rbf', 'matern1', 'matern3', 'matern5', 'linear', 'polynomial', 'polynomial_2',
                                           'rbf_linear', 'matern1_linear', 'matern3_linear', 'matern5_linear',
                                           'rbf_polynomial', 'matern1_polynomial', 'matern3_polynomial', 'matern5_polynomial',
                                           'rbf_polynomial_2', 'matern1_polynomial_2', 'matern3_polynomial_2', 'matern5_polynomial_2']:

            raise ValueError('Invalid kernel function')

    def _get_median(self, gamma_prior):
        k = gamma_prior.concentration
        r = gamma_prior.rate
        return (k - 1) / r

    def get_default_kernel(self):
        return self.add_scale_kernel(self.get_rbf_kernel())

    def get_specific_kernel(self):
        kernel_name = self.kernel_opt.lower()
        if kernel_name == 'rbf':
            return self.add_scale_kernel(self.get_rbf_kernel())

        if kernel_name == 'matern1':
            return self.add_scale_kernel(self.get_matern_kernel(nu=0.5))

        if kernel_name == 'matern3':
            return self.add_scale_kernel(self.get_matern_kernel(nu=1.5))

        if kernel_name == 'matern5':
            return self.add_scale_kernel(self.get_matern_kernel(nu=2.5))

        if kernel_name == 'linear':
            return self.add_scale_kernel(self.get_linear_kernel())

        if kernel_name == 'polynomial':
            return self.add_scale_kernel(self.get_polynomial_kernel())

        if kernel_name == 'polynomial_2':
            return self.add_scale_kernel(self.get_polynomial_kernel(power=2))

        if kernel_name == 'rbf_linear':
            return self.add_scale_kernel(self.get_rbf_kernel() + self.get_polynomial_kernel())

        if kernel_name == 'matern1_linear':
            return self.add_scale_kernel(self.get_matern_kernel(nu=0.5) + self.get_linear_kernel())

        if kernel_name == 'matern3_linear':
            return self.add_scale_kernel(self.get_matern_kernel(nu=1.5) + self.get_linear_kernel())

        if kernel_name == 'matern5_linear':
            return self.add_scale_kernel(self.get_matern_kernel(nu=2.5) + self.get_linear_kernel())

        if kernel_name == 'rbf_polynomial':
            return self.add_scale_kernel(self.get_rbf_kernel() + self.get_polynomial_kernel())

        if kernel_name == 'matern1_polynomial':
            return self.add_scale_kernel(self.get_matern_kernel(nu=0.5) + self.get_polynomial_kernel())

        if kernel_name == 'matern3_polynomial':
            return self.add_scale_kernel(self.get_matern_kernel(nu=1.5) + self.get_polynomial_kernel())

        if kernel_name == 'matern5_polynomial':
            return self.add_scale_kernel(self.get_matern_kernel(nu=2.5) + self.get_polynomial_kernel())

        if kernel_name == 'rbf_polynomial_2':
            return self.add_scale_kernel(self.get_rbf_kernel() + self.get_polynomial_kernel(power=2))

        if kernel_name == 'matern1_polynomial_2':
            return self.add_scale_kernel(self.get_matern_kernel(nu=0.5) + self.get_polynomial_kernel(power=2))

        if kernel_name == 'matern3_polynomial_2':
            return self.add_scale_kernel(self.get_matern_kernel(nu=1.5) + self.get_polynomial_kernel(power=2))

        if kernel_name == 'matern5_polynomial_2':
            return self.add_scale_kernel(self.get_matern_kernel(nu=2.5) + self.get_polynomial_kernel(power=2))

    # Selection of best kernel via cross validation
    def get_best_kernel(self):
        kernel_list = self.get_kernel_list()

        best_score = -float('inf')
        for idx, kernel in enumerate(kernel_list):
            kernel_dict = deepcopy(kernel.__dict__)

            score = self.cross_validation(kernel)

            kernel.__dict__ = kernel_dict

            if score > best_score:
                best_idx = idx
                best_score = score
                best_kernel = kernel

        return best_kernel

    def get_kernel_list(self):
        scale_rbf_kernel = self.add_scale_kernel(self.get_rbf_kernel())
        scale_mat_05_kernel = self.add_scale_kernel(self.get_matern_kernel(nu=0.5))
        scale_mat_15_kernel = self.add_scale_kernel(self.get_matern_kernel(nu=1.5))
        scale_mat_25_kernel = self.add_scale_kernel(self.get_matern_kernel(nu=2.5))
        scale_linear_kernel = self.add_scale_kernel(self.get_linear_kernel())
        scale_pol_kernel = self.add_scale_kernel(self.get_polynomial_kernel())
        scale_pol_2_kernel = self.add_scale_kernel(self.get_polynomial_kernel(power=2))

        scale_rbf_lin_kernel = self.add_scale_kernel(self.get_rbf_kernel() + self.get_polynomial_kernel())
        scale_mat_05_lin_kernel = self.add_scale_kernel(self.get_matern_kernel(nu=0.5) + self.get_linear_kernel())
        scale_mat_15_lin_kernel = self.add_scale_kernel(self.get_matern_kernel(nu=1.5) + self.get_linear_kernel())
        scale_mat_25_lin_kernel = self.add_scale_kernel(self.get_matern_kernel(nu=2.5) + self.get_linear_kernel())

        scale_rbf_pol_kernel = self.add_scale_kernel(self.get_rbf_kernel() + self.get_polynomial_kernel())
        scale_mat_05_pol_kernel = self.add_scale_kernel(self.get_matern_kernel(nu=0.5) + self.get_polynomial_kernel())
        scale_mat_15_pol_kernel = self.add_scale_kernel(self.get_matern_kernel(nu=1.5) + self.get_polynomial_kernel())
        scale_mat_25_pol_kernel = self.add_scale_kernel(self.get_matern_kernel(nu=2.5) + self.get_polynomial_kernel())

        scale_rbf_pol_2_kernel = self.add_scale_kernel(self.get_rbf_kernel() + self.get_polynomial_kernel(power=2))
        scale_mat_05_pol_2_kernel = self.add_scale_kernel(self.get_matern_kernel(nu=0.5) + self.get_polynomial_kernel(power=2))
        scale_mat_15_pol_2_kernel = self.add_scale_kernel(self.get_matern_kernel(nu=1.5) + self.get_polynomial_kernel(power=2))
        scale_mat_25_pol_2_kernel = self.add_scale_kernel(self.get_matern_kernel(nu=2.5) + self.get_polynomial_kernel(power=2))

        kernel_list = [scale_rbf_kernel,
                       scale_mat_05_kernel,
                       scale_mat_15_kernel,
                       scale_mat_25_kernel,
                       scale_linear_kernel,
                       scale_pol_kernel,
                       scale_pol_2_kernel,
                       scale_rbf_lin_kernel,
                       scale_mat_05_lin_kernel,
                       scale_mat_15_lin_kernel,
                       scale_mat_25_lin_kernel,
                       scale_rbf_pol_kernel,
                       scale_mat_05_pol_kernel,
                       scale_mat_15_pol_kernel,
                       scale_mat_25_pol_kernel,
                       scale_rbf_pol_2_kernel,
                       scale_mat_05_pol_2_kernel,
                       scale_mat_15_pol_2_kernel,
                       scale_mat_25_pol_2_kernel]

        return kernel_list

    def get_rbf_kernel(self):
        if self.ard:
            rbf_kernel = RBFKernel(ard_num_dims=self.num_dims,
                                   lengthscale_prior=self.lengthscale_prior)
        else:
            rbf_kernel = RBFKernel(lengthscale_prior=self.lengthscale_prior)

        if self.lengthscale_prior is not None:
            ls_median = self._get_median(self.lengthscale_prior)
            rbf_kernel.lengthscale = to_tensor(ls_median, gpu=self.gpu)

        return rbf_kernel

    def get_matern_kernel(self, nu=2.5):
        if self.ard:
            matern_kernel = MaternKernel(nu=nu,
                                         ard_num_dims=self.num_dims,
                                         lengthscale_prior=self.lengthscale_prior)
        else:
            matern_kernel = MaternKernel(nu=nu,
                                         lengthscale_prior=self.lengthscale_prior)

        if self.lengthscale_prior is not None:
            ls_median = self._get_median(self.lengthscale_prior)
            matern_kernel.lengthscale = to_tensor(ls_median, gpu=self.gpu)

        return matern_kernel

    def get_linear_kernel(self):
        linear_kernel = LinearKernel(variance_prior=self.variance_prior)

        if self.variance_prior is not None:
            var_median = self._get_median(self.variance_prior)
            linear_kernel.variance = to_tensor(var_median, gpu=self.gpu)

        return linear_kernel

    def get_polynomial_kernel(self, power=1):
        polynomial_kernel = PolynomialKernel(power=power,
                                             offset_prior=self.offset_prior)

        if self.offset_prior is not None:
            off_median = self._get_median(self.offset_prior)
            polynomial_kernel.offset = to_tensor(off_median, gpu=self.gpu)

        return polynomial_kernel

    def add_scale_kernel(self, base_kernel):
        kernel = ScaleKernel(base_kernel,
                             outputscale_prior=self.outputscale_prior)

        if self.outputscale_prior is not None:
            os_median = self._get_median(self.outputscale_prior)
            kernel.outputscale = to_tensor(os_median, gpu=self.gpu)

        # outputscaleに制約を課す
        kernel.register_constraint('raw_outputscale', GreaterThan(1e-2))

        return kernel

    def cross_validation(self, kernel, seed=0):
        kfold = KFold(n_splits=5, shuffle=True, random_state=seed)

        y_true, y_pred = [], []
        for train_idx, test_idx in kfold.split(self.X, self.y):
            cv_train_x = self.X[train_idx]
            cv_train_y = self.y[train_idx]
            cv_test_x = self.X[test_idx]
            cv_test_y = self.y[test_idx]

            cv_pred_y = self.gpr(cv_train_x, cv_train_y, cv_test_x, kernel)

            y_true += cv_test_y.detach().cpu().numpy().tolist()
            y_pred += cv_pred_y.tolist()

        r2 = r2_score(y_true, y_pred)

        return r2

    def gpr(self, train_x, train_y, test_x, kernel, n_restarts=0):
        surrogate_model = GPModel(train_x,
                                  train_y,
                                  kernel,
                                  gpu=self.gpu,
                                  n_restarts=n_restarts,
                                  noise_prior=self.noise_prior,
                                  noise_constraint=self.noise_constraint)

        surrogate_model.fit()
        pred_y = surrogate_model.predict_mean(test_x)

        return pred_y
