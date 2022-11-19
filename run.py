import pdb

import pandas as pd

from bo.bayesian_optimization import ReactionOpt


def run(seed=None):
    desc_df = pd.read_csv('descriptor.csv')
    descritor = [desc_df]

    """
    Parameters
    ----------
    input_name : str
        Name of input file (xxx of xxx.com).
    desc_data : list
        List of pandas.DataFrame of descriptors.
    init_method : str
        Strategy for selecting initial points for evaluation.
    maxtrial : int
        Maximum number of trials for virtual experiments.
    finish_thresh : float
        Convergence criteria for BO.
        Value of the 10th smallest barrier in the Claisen barrier dataset. 
    target : str
        Column label of objective variable.
    opt_type : 'minimize' or 'maximize'
        Specify whether to minimize or maximize the objective variable
    ard : bool
        Use ARD kernel
    gpu : bool
        Use GPUs (if available) to run gaussian process computations.
    batch_size : int
        Number of experiments selected via acquisition and initialization functions.
    batch_magnification : int
        The number of degrees of the computational batch size.
        If batch_size = x and batch_maginification = y, the computational batch size = x × y.
    second_opt : bool
        Perform Bayesian optimization using also computational results.
        If second_opt = False, BO is performed using only the virtual experimental results.
        If second_opt = True, BO is performed using virtual experimental results and
        virtual computational results.
    target_scaling : 'standard' or 'minmax'
        If target_scaling = 'standard', barrier data are standardized to have a mean of 0
        and a standard deviation of 1.
        If target_scaling = 'minmax', barrier data are normalized to have a minimum of 0
        and a maximum of 1.
    deviation : [str, float, float, (float)]
        Specify noise type and size.
        Options: ['normal', loc(mean), scale(standard deviation)]
                 ['gamma', shape, scale, loc]
                 ['uniform', left, right]
    """
    bo = ReactionOpt(input_name='read', desc_data=descritor,
                     init_method='rand', maxtrial=100,
                     finish_thresh=-3.4, target='Barrier',
                     opt_type='minimize', ard=False, gpu=False,
                     batch_size=5, batch_magnification=1, second_opt=True,
                     target_scaling='standard', deviation=['normal', 0, 5])

    if seed is None:
        bo.init_sample(seed=0)
    else:
        bo.init_sample(seed=seed)

    bo.run()
    bo.save()
    # bo.history()

    best_val = bo.get_best_result()
    total_trial = len(bo.real_row_results)

    return best_val, total_trial


if __name__ == '__main__':
    run()
