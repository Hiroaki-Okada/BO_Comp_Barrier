import pdb

import pandas as pd

from bo.pd_utils import to_tensor

class Objective:
    def __init__(self, results=pd.DataFrame(), row_results=pd.DataFrame(),
                 domain=pd.DataFrame(), mini_domain=pd.DataFrame(),
                 all_name_combs=pd.DataFrame(),
                 target='barrier', target_scaling='minmax',
                 opt_type='minimize', gpu=False):

        self.row_results = row_results
        self.domain = domain
        self.mini_domain = mini_domain
        self.all_name_combs = all_name_combs
        self.target = target
        self.target_scaling = target_scaling
        self.opt_type = opt_type
        self.gpu = gpu

        if target_scaling == 'standard':
            self.results = self.standardize(results)
        elif target_scaling == 'minmax':
            self.results = self.minmax(results)
        else:
            raise ValueError('Inappropriate scaling method')

        if len(self.results) > 0:
            self.X = to_tensor(self.results.drop(columns=target), gpu=gpu)
            self.y = to_tensor(self.results[target], gpu=gpu).view(-1)
        else:
            self.X = to_tensor([], gpu=gpu)
            self.y = to_tensor([], gpu=gpu)

    def set_mini_domain(self, mini_domain=pd.DataFrame()):
        self.mini_domain = mini_domain

    def clear_results(self):
        self.results = pd.DataFrame()
        self.X = to_tensor([], gpu=self.gpu)
        self.y = to_tensor([], gpu=self.gpu)

    def standardize(self, df):
        if len(df) == 0:
            return df

        unstandard_vector = df[self.target].values

        self.mean = unstandard_vector.mean()
        self.std = unstandard_vector.std()

        # Prevent divide by zero error
        if self.std == 0.0:
            self.std = 1e-6

        if len(df) == 1:
            return df

        standard_vector = (unstandard_vector - self.mean) / self.std

        new_df = df.copy().drop(columns=self.target)
        new_df[self.target] = standard_vector

        return new_df

    def minmax(self, df):
        if len(df) == 0:
            return df

        unminmax_vector = df[self.target].values

        self.max_t = unminmax_vector.max()
        self.min_t = unminmax_vector.min()

        if len(df) == 1:
            return df

        minmax_vector = (unminmax_vector - self.min_t) / (self.max_t - self.min_t)

        new_df = df.copy().drop(columns=self.target)
        new_df[self.target] = minmax_vector

        return new_df
