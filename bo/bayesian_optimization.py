import pdb

import os
import sys
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from gpytorch.priors import GammaPrior

from bo.surrogate_models import GPModel
from bo.exact_gp_model  import fast_computation
from bo.objective import Objective
from bo.acq_func import AcquisitionFunc
from bo.init_sample import InitSample
from bo.domain_utils import DomainUtils
from bo.desc_utils import modify_index, random_forest
from bo.kernel_utils import KernelUtils
from bo.status_utils import check_status
from bo.first_opt import FirstBO
from bo.second_opt import SecondBO
from bo.base_utils import ReadDataset

np.random.seed(seed=0)
torch.manual_seed(0)


class BayesianOptimization:
    def __init__(self,
                 input_name='read',
                 domain=pd.DataFrame(),
                 all_name_combs=pd.DataFrame(),
                 pre_real_results=pd.DataFrame(),
                 pre_calc_results=pd.DataFrame(),
                 real_row_results=pd.DataFrame(),
                 calc_row_results=pd.DataFrame(),
                 init_method='rand', fast_comp=True,
                 surrogate_model=GPModel, ard=False,
                 gpu=False, kernel_opt=False,
                 duplicate=False, acquisition_function='EI',
                 batch_size=5, batch_magnification=1,
                 target='barrier', target_scaling='minmax',
                 one_step=False, opt_type='minimize',
                 maxtrial=100, finish_thresh=None,
                 feature_selection=False, deviation=None,
                 mode='calculation', second_opt=False,
                 lengthscale_prior=GammaPrior(1.5, 0.1),
                 outputscale_prior=GammaPrior(7.0, 1.0),
                 variance_prior=GammaPrior(1.05, 0.5),
                 offset_prior=GammaPrior(1.05, 0.5),
                 noise_prior=GammaPrior(1.5, 0.5),
                 noise_constraint=1e-5):

        fast_computation(fast_comp)

        self.init_seq = InitSample(init_method, batch_size)

        self.obj = Objective(domain=domain,
                             all_name_combs=all_name_combs,
                             target=target,
                             target_scaling=target_scaling,
                             opt_type=opt_type,
                             gpu=gpu)

        self.kernel_utils = KernelUtils(kernel_opt=kernel_opt,
                                        ard=ard,
                                        gpu=gpu,
                                        lengthscale_prior=lengthscale_prior,
                                        outputscale_prior=outputscale_prior,
                                        variance_prior=variance_prior,
                                        offset_prior=offset_prior,
                                        noise_prior=noise_prior,
                                        noise_constraint=noise_constraint)

        self.acq = AcquisitionFunc(acquisition_function,
                                   batch_size=batch_size,
                                   duplicate=duplicate)

        self.read_dataset = ReadDataset(all_name_combs=all_name_combs,
                                        opt_type=opt_type,
                                        target=target,
                                        finish_thresh=finish_thresh)

        self.input_name = input_name
        self.domain = domain
        self.all_name_combs = all_name_combs
        self.pre_real_results = pre_real_results
        self.pre_calc_results = pre_calc_results
        self.real_row_results = real_row_results
        self.calc_row_results = calc_row_results
        self.init_method = init_method
        self.surrogate_model = surrogate_model
        self.ard = ard
        self.gpu = gpu
        self.kernel_opt = kernel_opt
        self.duplicate = duplicate
        self.acquisition_function = acquisition_function
        self.batch_size = batch_size
        self.batch_magnification = batch_magnification
        self.target = target
        self.target_scaling = target_scaling
        self.one_step = one_step
        self.opt_type = opt_type
        self.maxtrial = maxtrial
        self.finish_thresh = finish_thresh
        self.feature_selection = feature_selection
        self.deviation = deviation
        self.mode = mode
        self.second_opt = second_opt
        self.noise_prior = noise_prior
        self.noise_constraint = noise_constraint

        check_status(self)

    def init_sample(self, seed=None):
        self.obj.clear_results()
        if self.init_seq.method == 'read':
            pre_real_idx = modify_index(self.pre_real_results).index.values
            self.init_descs = self.obj.domain.loc[pre_real_idx, :]

            if len(self.pre_calc_results) > 0:
                pre_calc_idx = modify_index(self.pre_calc_results).index.values
                self.init_calc_descs = self.obj.domain.loc[pre_calc_idx, :]
        else:
            self.init_descs = self.init_seq.run(self.obj, seed=seed)

        return self.init_descs

    def init_process(self):
        if self.init_seq.method == 'read':
            pre_real_target = self.get_target(self.pre_real_results)
            self.real_row_results = pd.concat([self.init_descs,
                                               pre_real_target], axis=1)

            if len(self.pre_calc_results) > 0:
                pre_calc_target = self.get_target(self.pre_calc_results)
                self.calc_row_results = pd.concat([self.init_calc_descs,
                                                   pre_calc_target], axis=1)
        elif self.one_step:
            self.finish(self.init_descs)
        else:
            self.real_row_results = self.evaluate(self.init_descs)
            if self.second_opt:
                self.calc_row_results = self.evaluate(self.init_descs,
                                                      deviation=self.deviation)

    def run(self):
        self.init_process()

        bo_itr = 0
        t_start = time.time()
        while len(self.real_row_results) < self.maxtrial:
            bo_itr += 1
            self.bo(bo_itr)

            elapsed_time = round(time.time() - t_start, 1)
            print('\nBO-ITR:', bo_itr)
            print('Elapsed time =', elapsed_time, 'sec')

            if self.finish_thresh is not None:
                best = self.get_best_result()
                if self.opt_type == 'minimize' and best <= self.finish_thresh:
                    print('\nThe best value is below the threshold. Exit.')
                    df, _ = self.summarize(self.real_row_results)
                    print(df)
                    break
                if self.opt_type == 'maximize' and best >= self.finish_thresh:
                    print('\nThe best value is above the threshold. Exit.')
                    df, _ = self.summarize(self.real_row_results)
                    print(df)
                    break
        else:
            print('Maximum trial number was exceeded. Exit.')
            df, _ = self.summarize(self.real_row_results)
            print(df)

    def bo(self, bo_itr):
        if self.calc_row_results is not None:
            idx_1 = self.calc_row_results.index.values
            idx_2 = self.real_row_results.index.values
            unique_idx = list(np.isin(idx_1, idx_2, invert=True))
            unique_calc_results = self.calc_row_results[unique_idx]
            known_results = pd.concat([self.real_row_results, unique_calc_results])
        else:
            known_results = self.real_row_results.copy()

        self.obj.row_results = known_results

        if self.feature_selection:
            if bo_itr >= 2:
                selected_desc_names = random_forest(self.obj.row_results, target=self.target).tolist()
                mini_domain = self.obj.domain.loc[:, selected_desc_names]
            else:
                mini_domain = self.obj.domain.copy()
        else:
            mini_domain = self.obj.domain.copy()

        self.obj.set_mini_domain(mini_domain)

        if self.mode.lower() == 'calculation':
            proposed_descs = self.run_bo()
        elif self.mode.lower() == 'experiment':
            self.second_obj = deepcopy(self.obj)
            proposed_descs = self.run_sec_bo()

        if self.one_step:
            self.finish(proposed_descs)

        if self.second_opt:
            idx_1 = proposed_descs.index.values
            idx_2 = self.calc_row_results.index.values
            unique_idx = list(np.isin(idx_1, idx_2, invert=True))
            unique_descs = proposed_descs[unique_idx]

            self.calc_row_results = self.evaluate(unique_descs,
                                                  concat_df=self.calc_row_results,
                                                  deviation=self.deviation)

            self.second_obj = deepcopy(self.obj)
            self.second_obj.row_results = self.add_calc_results()

            proposed_descs = self.run_sec_bo()

        self.real_row_results = self.evaluate(proposed_descs,
                                              concat_df=self.real_row_results)

    # Propose computational batch
    def run_bo(self):
        first_bo = FirstBO(obj=self.obj,
                           kernel_utils=self.kernel_utils,
                           surrogate_model=self.surrogate_model,
                           ard=self.ard,
                           acq=self.acq,
                           batch_size=self.batch_size,
                           batch_magnification=self.batch_magnification,
                           noise_prior=self.noise_prior,
                           noise_constraint=self.noise_constraint)

        proposed_idx = first_bo.run()
        proposed_descs = self.obj.domain.loc[proposed_idx, :]

        return proposed_descs

    # Propose experimental batch
    def run_sec_bo(self):
        second_bo = SecondBO(obj=self.second_obj,
                             kernel_utils=self.kernel_utils,
                             real_row_results=self.real_row_results,
                             surrogate_model=self.surrogate_model,
                             ard=self.ard,
                             acq=self.acq,
                             deviation=self.deviation,
                             batch_size=self.batch_size,
                             noise_prior=self.noise_prior,
                             noise_constraint=self.noise_constraint)

        proposed_idx = second_bo.run()
        proposed_descs = self.obj.domain.loc[proposed_idx, :]

        return proposed_descs

    def add_calc_results(self):
        idx_1 = self.calc_row_results.index.values
        idx_2 = self.real_row_results.index.values
        unique_idx = list(np.isin(idx_1, idx_2, invert=True))
        unique_calc_results = self.calc_row_results[unique_idx]

        if len(self.real_row_results) > 0:
            row_results = pd.concat([self.real_row_results, unique_calc_results])

        return row_results


class ReactionOpt(BayesianOptimization):
    def __init__(self, input_name, desc_data,
                 other_components={}, encoding={},
                 pre_real_results=pd.DataFrame(),
                 pre_calc_results=pd.DataFrame(),
                 init_method='rand', surrogate_model=GPModel,
                 ard=False, gpu=False, kernel_opt=False,
                 duplicate=False, acquisition_function='EI',
                 batch_size=5, batch_magnification=1,
                 target='barrier', target_scaling='minmax',
                 one_step=False, opt_type='minimize',
                 maxtrial=100, finish_thresh=None,
                 feature_selection=False, deviation=None,
                 mode='calculation', second_opt=False):

        domain_utils = DomainUtils(input_name=input_name,
                                   desc_data=desc_data,
                                   other_components=other_components,
                                   encoding=encoding)

        domain, all_name_combs = domain_utils.run()

        super(ReactionOpt, self).__init__(input_name=input_name,
                                          domain=domain,
                                          all_name_combs=all_name_combs,
                                          pre_real_results=pre_real_results,
                                          pre_calc_results=pre_calc_results,
                                          init_method=init_method,
                                          surrogate_model=surrogate_model,
                                          ard=ard,
                                          kernel_opt=kernel_opt,
                                          duplicate=duplicate,
                                          acquisition_function=acquisition_function,
                                          batch_size=batch_size,
                                          batch_magnification=batch_magnification,
                                          target=target,
                                          target_scaling=target_scaling,
                                          one_step=one_step,
                                          opt_type=opt_type,
                                          maxtrial=maxtrial,
                                          finish_thresh=finish_thresh,
                                          feature_selection=feature_selection,
                                          deviation=deviation,
                                          mode=mode,
                                          second_opt=second_opt)

    # Evaluate reactivity (read barrier from dataset)
    def evaluate(self, descs, concat_df=None, deviation=None):
        row_results = self.read_dataset.get_row_results(descs,
                                                        concat_df=concat_df,
                                                        deviation=deviation)

        return row_results

    def get_target(self, df):
        target_vals = df.loc[:, self.target].copy()
        if self.opt_type == 'minimize':
            target_vals *= -1

        return target_vals

    def get_best_result(self):
        target_vals = self.get_target(self.real_row_results)

        sorted_target_vals = np.sort(target_vals)

        if self.opt_type == 'minimize':
            best = sorted_target_vals[0]
        if self.opt_type == 'maximize':
            best = sorted_target_vals[-1]

        return best

    def save(self):
        if os.path.isdir('results') == False:
            os.makedirs('results')

        summary_df, sorted_summary_df = self.summarize(self.real_row_results)
        summary_df.to_csv('results/experimental_results.csv')
        sorted_summary_df.to_csv('results/experimental_results_sorted.csv')

        if self.second_opt:
            calc_df, sorted_calc_df = self.summarize(self.calc_row_results)
            calc_df.to_csv('results/calculation_results.csv')
            sorted_calc_df.to_csv('results/calculation_results_sorted.csv')

    def summarize(self, results_df):
        if len(results_df) == 0:
            return results_df, results_df

        idx = results_df.index.values
        name_combs = self.all_name_combs.loc[idx, :]

        target_vals = self.get_target(results_df)
        df = pd.concat([name_combs, target_vals], axis=1)

        if self.opt_type == 'maximize':
            sorted_df = df.sort_values(by=self.target, ascending=False)
        elif self.opt_type == 'minimize':
            sorted_df = df.sort_values(by=self.target)
        else:
            raise ValueError('Invalid optimize mode')

        sorted_df.reset_index(drop=True)

        return df, sorted_df

    def history(self):
        self.history_plot(self.real_row_results)
        if self.second_opt:
            self.history_plot(self.calc_row_results, name='History_calc')

    def history_plot(self, results_df, name='History'):
        y = results_df.loc[:, self.target].values
        batch_y = np.reshape(y, (len(y) // self.batch_size, self.batch_size))

        if self.opt_type == 'minimize':
            batch_y = -batch_y

        batch_best_y = []
        if self.opt_type == 'minimize':
            best = float('inf')
            for i in batch_y:
                best = min(best, min(i))
                batch_best_y.append(best)
        else:
            best = -float('inf')
            for i in batch_y:
                best = max(best, max(i))
                batch_best_y.append(best)

        batch_x = [i for i in range(1, len(batch_best_y) + 1)]
        batch_x_ex = sum([[i] * self.batch_size for i in batch_x], [])

        plt.figure(figsize=(10, 10))
        plt.scatter(batch_x_ex, batch_y, s=200, alpha=0.8, c='darkgrey')
        plt.plot(batch_x, batch_best_y, lw=3, color='black')
        plt.scatter(batch_x, batch_best_y, s=200, c='black')
        plt.xlabel('Trial batch', fontsize=30)
        plt.ylabel(self.target, fontsize=30)
        plt.tick_params(labelsize=30)
        plt.tight_layout()
        plt.savefig(name + '.jpeg')

    # Stop optimization when one_step option=True
    def finish(self, descs):
        idx = descs.index.values
        name_combs = self.all_name_combs.loc[idx, :]

        next_target = pd.DataFrame([['<Enter ' + self.target + '>']] * len(idx),
                                   columns=[self.target], index=idx)

        next_batch = pd.concat([name_combs, next_target], axis=1)

        real_summary, _ = self.summarize(self.real_row_results)
        calc_summary, _ = self.summarize(self.calc_row_results)

        if self.mode.lower() == 'experiment':
            real_summary = pd.concat([real_summary, next_batch])
        elif self.mode.lower() == 'calculation':
            calc_summary = pd.concat([calc_summary, next_batch])
        else:
            raise ValueError('Invalid mode')

        if os.path.isdir('results') == False:
            os.makedirs('results')

        real_summary.to_csv('results/current_experimental_results.csv')
        calc_summary.to_csv('results/current_calculation_results.csv')

        print('One step option was set to True. Optimization was stopped.')
        print(real_summary)
        print(calc_summary)
        print(real_summary.shape, calc_summary.shape)

        sys.exit(0)
