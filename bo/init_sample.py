import pdb

import random
import numpy as np
import pandas as pd


class InitSample:
    def __init__(self, method, batch_size):
        self.method = method.lower()
        self.batch_size = batch_size

    def run(self, obj, seed=None, visualize=False):
        if self.method == 'rand':
            descs = rand(obj, self.batch_size, seed=seed)
        elif self.method == 'greedy':
            descs = greedy(obj, self.batch_size, seed=seed)
        elif self.method == 'read':
            descs = pd.DataFrame()
        else:
            raise ValueError('Invalid sampling method')

        return descs


def rand(obj, batch_size, seed=None):
    batch = obj.domain.sample(n=batch_size, random_state=seed)

    return batch


def greedy(obj, batch_size, seed=None):
    ini_x = obj.domain.sample(n=1, random_state=seed)
    np_batch = [ini_x.values[0]]
    np_domain = obj.domain.values
    batch_idx = ini_x.index.values.tolist()

    for i in range(batch_size - 1):
        max_dist = 0
        for idx, x in enumerate(np_domain):
            sum_dist = 0
            for selected in np_batch:
                sum_dist += np.linalg.norm(x - selected)

            if sum_dist > max_dist:
                best_idx = idx
                max_dist = sum_dist

        batch_idx.append(best_idx)
        np_batch.append(np_domain[best_idx])

    batch = obj.domain.loc[batch_idx, :]

    return batch
