import pdb

import numpy as np

import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan

from bo.exact_gp_model import ExactGPModel
from bo.opt_utils import optimize_mll
from bo.pd_utils import to_tensor

# Gaussian Process Model
class GPModel:
    def __init__(self, X, y, kernel, gpu=False,
                 n_restarts=0, noise_prior=None, noise_constraint=1e-5):

        self.X = X
        self.y = y
        self.gpu = gpu
        self.n_restarts = n_restarts

        self.likelihood = GaussianLikelihood()

        if noise_prior is not None:
            k = noise_prior.concentration
            r = noise_prior.rate
            noise_median = (k - 1) / r

            self.likelihood = GaussianLikelihood(noise_prior=noise_prior)
            self.likelihood.noise = to_tensor(noise_median, gpu=gpu)

        self.gp_model = ExactGPModel(self.X, self.y, self.likelihood, kernel=kernel)

        self.gp_model.likelihood.noise_covar.register_constraint('raw_noise',
                                                                 GreaterThan(noise_constraint))

        if torch.cuda.is_available() and gpu == True:
            self.gp_model = self.gp_model.cuda()

    # Maximum likelihood estimation
    def fit(self):
        min_loss_l = optimize_mll(self.gp_model,
                                  self.likelihood,
                                  self.X,
                                  self.y,
                                  n_restarts=self.n_restarts)

        self.min_loss_l = min_loss_l

    # Mean of predictive posterior
    def predict_mean(self, domain):
        self.gp_model.eval()
        self.likelihood.eval()

        domain = to_tensor(domain, gpu=self.gpu)

        f_preds = self.gp_model(domain)
        f_mean = f_preds.mean.detach()

        if torch.cuda.is_available() and self.gpu:
            f_mean = f_mean.cpu()

        f_mean = f_mean.numpy()

        return f_mean

    # GP prediction variance
    def get_variance(self, domain):
        self.gp_model.eval()
        self.likelihood.eval()

        domain = to_tensor(domain, gpu=self.gpu)

        f_preds = self.gp_model(domain)
        f_variance = f_preds.variance.detach()

        if torch.cuda.is_available() and self.gpu:
            f_variance = f_variance.cpu()

        f_variance = f_variance.numpy()

        return f_variance

    # Sample posterior
    def sample_posterior(self, domain, batch_size=1):
        self.gp_model.eval()
        self.likelihood.eval()

        domain = to_tensor(domain, gpu=self.gpu)

        f_preds = self.gp_model(domain)
        samples = f_preds.sample(torch.Size([batch_size])).detach()


        if torch.cuda.is_available() and self.gpu:
            samples = samples.cpu()

        samples = samples.numpy()

        return samples
