import pdb
import sys

import numpy as np
import pandas as pd

import torch

def to_tensor(data, gpu=False):
    if type(data) is torch.Tensor:
        return data

    try:
        tensor_data = torch.from_numpy(np.array(data).astype('float')).float()
    except:
        tensor_data = torch.tensor(data).float()

    if torch.cuda.is_available() == gpu == True:
        tensor_data = tensor_data.cuda()

    return tensor_data


def complement(df1, known_idx):
    df1_idx = df1.index.values
    boolean = list(np.isin(df1_idx, known_idx, invert=True))

    return boolean


def pd_argmax(sample_x_y, known_idx, val='acq_val', duplicate=False, top_n=1):
    if duplicate:
        sorted_sample = sample_x_y.sort_values(by=val, ascending=False)
        arg_max = sorted_sample.iloc[0:top_n]
    else:
        keep = complement(sample_x_y, known_idx)
        unique_sample_x_y = sample_x_y[keep]
        sorted_sample_x_y = unique_sample_x_y.sort_values(by=val, ascending=False)
        arg_max = sorted_sample_x_y.iloc[0:top_n]

    return arg_max


# Imputation of missing values by random forest regression
def rf_imputation(data, target='barrier', opt_type='minimize', finish_thresh=None):
    from sklearn.ensemble import RandomForestRegressor

    data_columns = data.columns

    ok_bool = data.loc[:, target].notnull()
    ng_bool = data.loc[:, target].isnull()

    ok_data = data[ok_bool]
    ng_data = data[ng_bool]

    if len(ng_data.index) == 0:
        return data

    X_train = ok_data.drop(columns=target)
    X_test = ng_data.drop(columns=target)
    y_train = ok_data[target]

    train_index = X_train.index.values
    test_index = X_test.index.values

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values

    rf = RandomForestRegressor()

    try:
        rf.fit(X_train, y_train)
    except:
        print('All data is NaN...exit')
        sys.exit(0)

    y_pred = rf.predict(X_test)

    if finish_thresh is not None:
        if opt_type == 'minimize':
            y_pred = np.array([min(-finish_thresh - 0.1, i) for i in y_pred])
        else:
            y_pred = np.array([min(finish_thresh - 0.1, i) for i in y_pred])

    X_train = pd.DataFrame(X_train, index=train_index)
    X_test = pd.DataFrame(X_test, index=test_index)

    y_train = pd.DataFrame(y_train, index=train_index)
    y_pred = pd.DataFrame(y_pred, index=test_index)

    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_pred])

    data = pd.concat([X, y], axis=1)
    data.columns = data_columns

    return data
