import pdb

import os
os.environ['OMP_NUM_THREADS'] = '1'

from copy import deepcopy

import numpy as np
import torch
import gpytorch


# Optimize a model via MLE
def optimize_mll(model, likelihood, train_X, train_y,
                 training_iters=200, learning_rate=0.2, n_restarts=0):

    model.train()
    likelihood.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=learning_rate)

    for loop in range(n_restarts + 1):
        for itr in range(training_iters):
            optimizer.zero_grad()

            output = model(train_X)
            loss = -mll(output, train_y)

            loss.backward()
            optimizer.step()

            if itr >= 1 and abs(loss - pref_loss) <= 1e-4:
                break

            pref_loss = loss.item()

        current_states = deepcopy(mll.model.state_dict())

        if loop == 0:
            min_loss_l = [loss.item()]
            states_l = [current_states]
        else:
            min_loss_l.append(loss.item())
            states_l.append(current_states)

    min_loss_idx = np.argmin(min_loss_l)
    mll.model.load_state_dict(states_l[min_loss_idx])

    return min_loss_l
