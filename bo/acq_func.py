import pdb

import math
import random
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import norm

from bo.pd_utils import to_tensor, complement, pd_argmax


class AcquisitionFunc:
    def __init__(self, function, batch_size=1, duplicate=False):
        function = function.lower()
        if function == 'ei':
            self.function = KrigingBeliever(EI, batch_size, duplicate)
        elif function == 'pi':
            self.function = KrigingBeliever(PI, batch_size, duplicate)
        elif function == 'ucb':
            self.function = KrigingBeliever(UCB, batch_size, duplicate)
        elif function == 'ei-rand':
            self.function = RandKrigingBeliever(EI, batch_size, duplicate)
        elif function == 'pi-rand':
            self.function = RandKrigingBeliever(PI, batch_size, duplicate)
        elif function == 'ucb-rand':
            self.function = RandKrigingBeliever(UCB, batch_size, duplicate)
        elif function == 'hybrid-ei':
            self.function = HybridThompsonSampling(EI, batch_size, duplicate)
        elif function == 'hybrid-pi':
            self.function = HybridThompsonSampling(PI, batch_size, duplicate)
        elif function == 'hybrid-ucb':
            self.function = HybridThompsonSampling(UCB, batch_size, duplicate)
        elif function == 'ts':
            self.function = ThompsonSampling(batch_size, duplicate)
        elif function == 'epsilon_greedy':
            self.function = EpsilonGreedy(batch_size, duplicate)
        elif function == 'random':
            self.function = RandomSampling(batch_size, duplicate)
        else:
            raise ValueError('Invalid acquisition function')

    def evaluate(self, surrogate_model, obj, kernel, calc_only_idx=None):
        proposed_idx = self.function.run(surrogate_model, obj, kernel, calc_only_idx=calc_only_idx)

        return proposed_idx


# Expected Improvement
def EI(surrogate_model, obj, jitter=0.01):
    if len(obj.results) == 0:
        max_observed = 0
    else:
        best_results = obj.results.sort_values(by=obj.target).iloc[-1]
        max_observed = best_results[obj.target]

    mean = surrogate_model.predict_mean(obj.mini_domain)
    variance = surrogate_model.get_variance(obj.mini_domain)

    isNegative = variance < 0
    if np.any(isNegative):
        warnings.warn('Predicted variances smaller than 0. '
                      'Setting those variances to 0.')
        variance = np.where(variance >= 0, variance, 0.00)

    # Prevent divide by zero error
    stdev = np.sqrt(variance) + 1e-6

    z = (mean - max_observed - jitter) / stdev
    imp = mean - max_observed - jitter

    ei = imp * norm.cdf(z) + stdev * norm.pdf(z)

    ei[stdev < jitter] = 0.0

    return ei


# Probability of Improvement
def PI(surrogate_model, obj, jitter=0.01):
    if len(obj.results) == 0:
        max_observed = 0
    else:
        best_results = obj.results.sort_values(by=obj.target).iloc[-1]
        max_observed = best_results[obj.target]

    mean = surrogate_model.predict_mean(obj.mini_domain)
    variance = surrogate_model.get_variance(obj.mini_domain)

    isNegative = variance < 0
    if np.any(isNegative):
        warnings.warn('Predicted variances smaller than 0. '
                      'Setting those variances to 0.')
        variance = np.where(variance >= 0, variance, 0.00)

    # Prevent divide by zero error
    stdev = np.sqrt(variance) + 1e-6

    z = (mean - max_observed - jitter) / stdev
    cdf = norm.cdf(z)

    cdf[stdev < jitter] = 0.0

    return cdf


# Upper Confidence Bound
def UCB(surrogate_model, obj, jitter=0.01, delta=0.5):
    mean = surrogate_model.predict_mean(obj.mini_domain)
    variance = surrogate_model.get_variance(obj.mini_domain)

    isNegative = variance < 0
    if np.any(isNegative):
        warnings.warn('Predicted variances smaller than 0. '
                      'Setting those variances to 0.')
        variance = np.where(variance >= 0, variance, 0.00)

    # Prevent divide by zero error
    stdev = np.sqrt(variance) + 1e-6

    dim = len(obj.domain.columns.values)
    iters = len(obj.results)

    beta = np.sqrt(2 * np.log(dim * (iters**2) * (math.pi**2) / (6*delta)))
    ucb = mean + beta * stdev

    return ucb


# Checking data to reject
def get_known_idx(obj, proposed, calc_only_idx=None, reject_idx=None):
    known_idx = obj.results.index.values

    if calc_only_idx is not None:
        unique_idx = list(np.isin(known_idx, calc_only_idx, invert=True))
        known_idx = known_idx[unique_idx]

    if reject_idx is not None:
        unique_idx = list(np.isin(reject_idx, known_idx, invert=True))
        reject_idx_cp = reject_idx.copy()[unique_idx]
        known_idx = np.concatenate([known_idx, reject_idx_cp], axis=0)

    proposed_idx = proposed.index.values.astype(known_idx.dtype)
    known_idx = np.concatenate([known_idx, proposed_idx], axis=0)

    return known_idx


class KrigingBeliever:
    def __init__(self, acq_function, batch_size, duplicate):
        self.acq_function = acq_function
        self.batch_size = batch_size
        self.duplicate = duplicate
        self.jitter = 0.01

    def run(self, surrogate_model, obj, kernel, calc_only_idx=None):
        surrogate_model_dict = surrogate_model.__dict__.copy()
        kernel_dict = deepcopy(kernel.__dict__)

        proposed = pd.DataFrame(columns=obj.mini_domain.columns)

        beliefs = []
        for i in range(self.batch_size):
            acq_val = self.acq_function(surrogate_model, obj, jitter=self.jitter)

            if self.duplicate == False:
                tmp_domain = obj.mini_domain.copy()
                tmp_domain['acq_val'] = acq_val

                known_idx = get_known_idx(obj, proposed, calc_only_idx=calc_only_idx)

                argmax_i = pd_argmax(tmp_domain, known_idx, duplicate=self.duplicate)
                proposed_i = argmax_i.drop(columns='acq_val')
            else:
                proposed_i = pd.DataFrame(data=obj.mini_domain.iloc[[np.argmax(acq_val)]])

            proposed = pd.concat([proposed, proposed_i], sort=False)

            mean_i = surrogate_model.predict_mean(proposed_i)
            beliefs.append(mean_i)

            cp_proposed = proposed.copy()
            cp_proposed[obj.target] = beliefs

            kriging_results = pd.concat([obj.results.copy(), cp_proposed], sort=False)

            train_X = to_tensor(kriging_results.drop(columns=obj.target), gpu=obj.gpu)
            train_y = to_tensor(kriging_results[obj.target], gpu=obj.gpu)

            surrogate_model.X = train_X
            surrogate_model.y = train_y

            surrogate_model.gp_model.train_inputs = (train_X,)
            surrogate_model.gp_model.train_targets = train_y

            surrogate_model.fit()

        surrogate_model.__init__(obj.X, obj.y, kernel)
        for key in surrogate_model_dict:
            surrogate_model.__dict__[key] = surrogate_model_dict.copy()[key]

        surrogate_model.gp_model.train_inputs = (obj.X,)
        surrogate_model.gp_model.train_targets = obj.y
        surrogate_model.fit()

        return proposed.index.values


class ThompsonSampling:
    def __init__(self, batch_size, duplicate):
        self.batch_size = batch_size
        self.duplicate = duplicate

    def run(self, surrogate_model, obj, kernel, calc_only_idx=None):
        samples = self.sample_posterior(surrogate_model, obj.mini_domain)

        proposed = pd.DataFrame(columns=obj.mini_domain.columns)
        for sample_i in samples:
            next_domain = obj.mini_domain.copy()
            next_domain['sample'] = sample_i

            known_idx = get_known_idx(obj, proposed, calc_only_idx=calc_only_idx)

            argmax_i = pd_argmax(next_domain, known_idx,
                                 val='sample', duplicate=self.duplicate)

            proposed_i = argmax_i.drop(columns='sample')
            proposed = pd.concat([proposed, proposed_i], sort=False)

        return proposed.index.values

    def sample_posterior(self, surrogate_model, domain, chunk_size=5000):
        if len(domain) < chunk_size:
            samples = surrogate_model.sample_posterior(domain, self.batch_size)
        else:
            samples = self.chunk_sample(surrogate_model, domain)

        return samples

    def chunk_sample(self, surrogate_model, domain, chunk_size=5000):
        chunks = len(domain) // chunk_size
        reminder = len(domain) % chunk_size

        samples = pd.DataFrame()
        for i in range(chunks):
            c_domain = domain[i*chunk_size:(i+1)*chunk_size]
            sample = surrogate_model.sample_posterior(c_domain, self.batch_size)
            sample = pd.DataFrame(sample)
            samples = pd.concat([samples, sample], axis=1)

        if reminder > 0:
            r_domain = domain[-reminder:]
            sample = surrogate_model.sample_posterior(r_domain, self.batch_size)
            sample = pd.DataFrame(sample)
            samples = pd.concat([samples, sample], axis=1)

        return samples.values


class EpsilonGreedy:
    def __init__(self, batch_size, duplicate):
        self.batch_size = batch_size
        self.duplicate = duplicate
        self.epsilon = 0.1

    def run(self, surrogate_model, obj, kernel, calc_only_idx=None):
        pred_mean = surrogate_model.predict_mean(obj.mini_domain)

        next_domain = obj.mini_domain.copy()
        next_domain['pred'] = pred_mean

        proposed = pd.DataFrame(columns=obj.mini_domain.columns)
        for i in range(self.batch_size) :
            known_idx = get_known_idx(obj, proposed, calc_only_idx=calc_only_idx)

            rand = random.random()
            if rand <= self.epsilon:
                if self.duplicate == False:
                    cand_domain = obj.domain[complement(obj.domain, known_idx)]
                else:
                    cand_domain = obj.domain

                proposed_i = cand_domain.sample(1)

            else:
                argmax_i = pd_argmax(next_domain, known_idx,
                                     val='pred', duplicate=self.duplicate)

                proposed_i = argmax_i.drop(columns='pred')

            proposed = pd.concat([proposed, proposed_i], sort=False)

        return proposed.index.values


class RandomSampling:
    def __init__(self, batch_size, duplicate):
        self.batch_size = batch_size
        self.duplicate = duplicate

    def run(self, surrogate_model, obj, kernel, calc_only_idx=None, reject_idx=None):
        proposed = pd.DataFrame(columns=obj.mini_domain.columns)
        known_idx = get_known_idx(obj, proposed, calc_only_idx=calc_only_idx, reject_idx=reject_idx)

        if self.duplicate == False:
            cand_domain = obj.domain[complement(obj.domain, known_idx)]
        else:
            cand_domain = obj.domain

        proposed = cand_domain.sample(self.batch_size)

        return proposed.index.values


class RandKrigingBeliever:
    def __init__(self, acq_function, batch_size, duplicate, alpha=3):
        rand_batch_size = (batch_size + alpha - 1 ) // alpha
        kb_batch_size = batch_size - rand_batch_size

        self.kriging_believer = KrigingBeliever(acq_function,
                                                kb_batch_size,
                                                duplicate)

        self.rand_sampling = RandomSampling(rand_batch_size, duplicate)

    def run(self, surrogate_model, obj, kernel, calc_only_idx=None):
        kb_proposed_idx = self.kriging_believer.run(surrogate_model, obj,
                                                    kernel, calc_only_idx)

        reject_idx = np.array(kb_proposed_idx)
        rand_proposed_idx = self.rand_sampling.run(surrogate_model, obj, kernel,
                                                   calc_only_idx, reject_idx)

        proposed_idx = np.concatenate([kb_proposed_idx, rand_proposed_idx], axis=0)

        return proposed_idx


class HybridThompsonSampling:
    def __init__(self, acq_function, batch_size, duplicate):
        self.acq_function = acq_function
        self.batch_size = batch_size
        self.duplicate = duplicate
        self.jitter = 0.01

    def run(self, surrogate_model, obj, kernel, calc_only_idx=None):
        acq_val = self.acq_function(surrogate_model, obj, jitter=self.jitter)
        proposed = pd.DataFrame(columns=obj.mini_domain.columns)

        if self.duplicate == False:
            tmp_domain = obj.mini_domain.copy()
            tmp_domain['acq_val'] = acq_val

            known_idx = get_known_idx(obj, proposed, calc_only_idx=calc_only_idx)

            argmax_i = pd_argmax(tmp_domain, known_idx, duplicate=self.duplicate)
            proposed_i = argmax_i.drop(columns='acq_val')
        else:
            proposed_i = pd.DataFrame(data=obj.mini_domain.iloc[[np.argmax(acq_val)]])

        proposed = pd.concat([proposed, proposed_i], sort=False)

        if self.batch_size > 1:
            samples = self.sample_posterior(surrogate_model, obj.mini_domain)

            for sample_i in samples:
                next_domain = obj.mini_domain.copy()
                next_domain['sample'] = sample_i

                known_idx = get_known_idx(obj, proposed, calc_only_idx=calc_only_idx)

                argmax_i = pd_argmax(next_domain, known_idx,
                                     val='sample', duplicate=self.duplicate)

                proposed_i = argmax_i.drop(columns='sample')
                proposed = pd.concat([proposed, proposed_i], sort=False)

        return proposed.index.values

    def sample_posterior(self, surrogate_model, domain, chunk_size=5000):
        if len(domain) < chunk_size:
            samples = surrogate_model.sample_posterior(domain, self.batch_size - 1)
        else:
            samples = self.chunk_sample(surrogate_model, domain)

        return samples

    def chunk_sample(self, surrogate_model, domain, chunk_size=5000):
        chunks = len(domain) // chunk_size
        reminder = len(domain) % chunk_size

        samples = pd.DataFrame()
        for i in range(chunks):
            c_domain = domain[i*chunk_size:(i+1)*chunk_size]
            sample = surrogate_model.sample_posterior(c_domain, self.batch_size - 1)
            sample = pd.DataFrame(sample)
            samples = pd.concat([samples, sample], axis=1)

        if reminder > 0:
            r_domain = domain[-reminder:]
            sample = surrogate_model.sample_posterior(r_domain, self.batch_size - 1)
            sample = pd.DataFrame(sample)
            samples = pd.concat([samples, sample], axis=1)

        return samples.values
