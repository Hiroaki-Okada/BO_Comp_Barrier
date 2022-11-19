import pdb

import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


class ReadDataset:
    def __init__(self, all_name_combs, opt_type='minimize', target='barrier', finish_thresh=None):
        """This class provides a set of functions to return reaction barriers from the
        Claisen barrier dataset for the substituent combinations proposed by Bayesian optimization.
        A virtual experiment returns the reaction barrier from the dataset as is, while
        a virtual computation returns the reaction barrier from the dataset plus noise. 
        Note that the pairs of substituent combinations and reaction barriers in the dataset are
        pre-converted to dictionary type and defined directly in dataset_dict/dataset_dict.py.
        """
        self.all_name_combs = all_name_combs
        self.opt_type = opt_type
        self.target = target
        self.finish_thresh = finish_thresh

        from bo.dataset_dict import dataset_dict
        self.dataset = dataset_dict

    def get_row_results(self, descs, concat_df=None, deviation=None):
        target_vals = self.get_dataset_value(descs, deviation=deviation)
        row_results = pd.concat([descs, target_vals], axis=1)

        if concat_df is not None:
            row_results = pd.concat([concat_df, row_results])

        if None in target_vals:
            warnings.warn('Missing values were contained in database. '
                          'These values are interpolated by random forest regression')

            from pd_utils import rf_imputation
            row_results = rf_imputation(row_results,
                                        target=self.target,
                                        opt_type=self.opt_type,
                                        finish_thresh=self.finish_thresh)

        return row_results

    def get_dataset_value(self, descs, deviation=None):
        idx = descs.index.values
        name_combs_l = self.all_name_combs.loc[idx, :].values.tolist()

        target_vals_l = self.read_dataset(name_combs_l, deviation=deviation)
        target_vals_df = pd.DataFrame(target_vals_l, index=idx, columns=[self.target])

        return target_vals_df

    def read_dataset(self, name_combs_l, deviation=None):
        target_vals_l = []
        for i in name_combs_l:
            name_str = ''.join([str(j) for j in i])

            if name_str in self.dataset.barrier_dict:
                val = self.dataset.barrier_dict[name_str]
            else:
                raise ValueError('Invalid input array')

            if val is not None and deviation is not None:
                prob_distribution = ProbabilityDistribution(deviation)
                deviation_sample = prob_distribution.run(mode='sampling')
                val += deviation_sample

            if val is not None and self.opt_type == 'minimize':
                val = -val

            target_vals_l.append(val)

        return target_vals_l


class ProbabilityDistribution:
    def __init__(self, settings):
        self.name = settings[0].lower()
        self.param = settings[1:]

        if self.name not in ['normal', 'gamma', 'uniform']:
            raise ValueError('Invalid distribution type')

    def run(self, mode):
        if mode not in ['sampling', 'visualization']:
            raise ValueError('Invalid operation type')

        if mode == 'sampling':
            return self.sampling()

        elif mode == 'visualization':
            self.visualization()

    def sampling(self):
        if self.name == 'normal':
            mean, stdev = self.extract()
            sample = stats.norm.rvs(loc=mean, scale=stdev, size=1)

        elif self.name == 'gamma':
            shape, scale, loc = self.extract()
            sample = stats.gamma.rvs(a=shape, scale=1/scale, loc=loc, size=1)

        elif self.name == 'uniform':
            left, right = self.extract()
            sample = stats.uniform.rvs(loc=left, scale=right-left, size=1)

        return sample.tolist()[0]

    def extract(self):
        if self.name == 'normal':
            try:
                mean, stdev = self.param[0], self.param[1]
            except:
                mean, stdev = 0.0, 1.0

            return mean, stdev

        elif self.name == 'gamma':
            try:
                shape, scale, loc = self.param[0], self.param[1], self.param[2]
            except:
                shape, scale, loc = 1.05, 0.1, 0.0

            return shape, scale, loc

        elif self.name == 'uniform':
            try:
                left, right = self.param[0], self.param[1]
            except:
                left, right = -1.0, 1.0

            return left, right

    def visualization(self):
        if self.name == 'normal':
            mean, stdev = self.extract()
            min_x = mean - stdev * 4
            max_x = mean + stdev * 4

        elif self.name == 'gamma':
            shape, scale, loc = self.extract()
            min_x = loc
            max_x = loc + 100

        elif self.name == 'uniform':
            left, right = self.extract()
            min_x = left
            max_x = right

        x = np.linspace(min_x, max_x, 500)

        if self.name == 'normal':
            y = stats.norm.pdf(x, loc=mean, scale=stdev)
        elif self.name == 'gamma':
            y = stats.gamma.pdf(x, a=shape, scale=1/scale, loc=loc)
        elif self.name == 'uniform':
            y = stats.uniform.pdf(x, loc=left, scale=right-left)

        plt.figure(figsize=(12, 8))
        plt.plot(x, y, alpha=0.6)
        plt.xlabel('x', fontsize=30)
        plt.ylabel('p(x|a,b)', fontsize=30)
        plt.tick_params(labelsize=20)
        plt.grid(True)

        if self.name == 'normal':
            plt.title('Normal distribution (μ=%.1f, σ=%.1f) pdf'
                      % (mean, stdev), fontsize=30)
            plt.xlim(min_x, max_x)

        elif self.name == 'gamma':
            plt.title('Gamma distribution (shape=%.1f, scale=%.1f, loc=%.1f) pdf'
                      % (shape, scale, loc), fontsize=30)
            plt.xlim(min_x, x[np.where(y > 1e-3)[0][-1]])

        elif self.name == 'uniform':
            plt.title('Uniform distribution (min=%.1f, max=%.1f) pdf'
                      % (left, right), fontsize=30)
            plt.xlim(min_x, max_x)

        plt.tight_layout()
        plt.show()


def plot_distribution():
    while True:
        print('\nEnter distribution type (normal, gamma, uniform)')
        name = input()

        if name == 'normal':
            print('\nEnter mean and standard deviation (like 0 1)')
            mean, stdev = map(float, input().split())
            deviation = [name, mean, stdev]

        elif name == 'gamma':
            print('\nEnter shape, scale, loc (like 1 1 0)')
            shape, scale, loc = map(float, input().split())
            deviation = [name, shape, scale, loc]

        elif name == 'uniform':
            print('\nEnter left, right (like -1 1)')
            left, right = map(float, input().split())
            deviation = [name, left, right]

        else:
            print('\nSpecify valid distribution type.')
            continue

        prob_distribution = ProbabilityDistribution(deviation)
        prob_distribution.run(mode='visualization')


if __name__ == '__main__':
    plot_distribution()
