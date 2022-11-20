import sys

import time
t_start = time.time()

import numpy as np
import matplotlib.pyplot as plt

import run


def bo_opt(ini_seed, loop_num, max_trial, batch_size, finish_thresh):
    half_trial = max_trial // 2
    success_count = [0 for i in range(max_trial + 1)]

    ok_cnt = 0
    over_cnt = 0
    trial_sum = 0
    for loop, seed in enumerate(range(ini_seed, ini_seed + loop_num)):
        elapsed_time = round(time.time() - t_start, 1)
        print('\n\nSeed:', seed)
        print('Total elapsed time =', elapsed_time, 'sec')

        best_val, total_trial = run.run(seed)

        if best_val <= finish_thresh:
            ok_cnt += 1
            for i in range(total_trial, max_trial + 1):
                success_count[i] += 1
        else:
            over_cnt += 1

        trial_sum += total_trial

        c_success_rate = [100 * i / (loop + 1) for i in success_count]

        print(half_trial, ' trials:', c_success_rate[half_trial], '%')
        print(max_trial, 'trials:', c_success_rate[max_trial], '%')

        sys.stdout.flush()

    print('\n\n*** Results of performance benchmark ***\n')
    print('Success time            :', ok_cnt, '/', loop_num)
    print('Excess time             :', over_cnt, '/', loop_num)
    print('Average number of trials:', round(trial_sum / loop_num, 1))

    success_rate = [round(100 * i / loop_num, 1) for i in success_count]

    print('\nSuccess rate summary')
    print(half_trial, ' trials:', success_rate[half_trial], '%')
    print(max_trial, 'trials:', success_rate[max_trial], '%')

    print('\nSuccess rate list')
    print(success_rate)
    success_rate = np.array(success_rate)

    for idx, rate in enumerate(success_rate):
        if rate >= 95:
            print('\nNumber of trials required to reach 95% rate of success:', idx, '\n')
            break

    sys.stdout.flush()

    return success_rate


def visualize(success_rate, max_trial, batch_size):
    x = [i for i in range(max_trial + 1)][::batch_size]
    y = success_rate[::batch_size]

    plt.figure(figsize=(20, 15))
    plt.plot(x, y, linewidth=5)
    plt.scatter(x, y, linewidths=7)
    plt.title('Performance', fontsize=50)
    plt.xlabel('Trials', fontsize=50)
    plt.ylabel('Success rate (%)', fontsize=50)
    plt.tick_params(labelsize=50)
    plt.ylim(-1, 101)
    plt.tight_layout()
    plt.savefig('Success_rate.jpeg')


def benchmark(ini_seed=0, loop_num=200, max_trial=100, batch_size=5, finish_thresh=-3.4):
    """Run performance benchmark program.

    Parameters
    ----------
    ini_seed : int
        Initial value of random number seed.
    loop_num : int
        Number of times to run Bayesian optimization program.
    max_trial : int
        Maximum number of trials for virtual experiments.
        Should be the same as the value of 'maxtrial' given to the ReactionOpt class in run.py
    batch_size : int
        Number of experiments selected via acquisition and initialization functions.
        Should be the same as the value of 'batch_size' given to the ReactionOpt class in run.py
    finish_thresh : float
        Convergence criteria for Bayesian optimization.
        Value of the 10th smallest barrier in the Claisen barrier dataset.
    """
    success_rate = bo_opt(ini_seed, loop_num, max_trial, batch_size, finish_thresh)
    visualize(success_rate, max_trial, batch_size)


if __name__ == '__main__':
    benchmark()
