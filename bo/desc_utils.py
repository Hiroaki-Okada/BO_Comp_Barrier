import pdb
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


def modify_index(data, name='Unnamed: 0'):
    cand_name = data.loc[:, name].values
    data.index = cand_name
    data.drop(columns=name, inplace=True)

    return data


def drop_single_value_columns(data):
    keep = []
    for i in range(len(data.columns)):
        set_len = len(data.iloc[:, i].drop_duplicates())
        if set_len > 1:
            keep.append(data.columns.values[i])

    return data[keep]


def drop_string_columns(data):
    keep = []
    for i in range(len(data.columns)):
        unique = data.iloc[:, i].drop_duplicates()
        isKeep = True
        for j in range(len(unique)):
            if type(unique.iloc[j]) is str:
                isKeep = False
                break

        if isKeep:
            keep.append(data.columns.values[i])

    return data[keep]


def drop_correlated_features(data, target=None, threshold=0.95):
    if target is not None:
        cp_data = data.drop(columns=target)
    else:
        cp_data = data.copy()

    corr = data.corr().abs()

    keep = []
    for i in range(len(corr.columns)):
        above = corr.iloc[:i, i]
        if len(keep) > 0:
            above = above[keep]

        isOk = above < threshold
        if len(above[isOk]) == len(above):
            keep.append(corr.columns.values[i])

    return data[keep]


def desc_cleaning(data):
    data = modify_index(data, name='Unnamed: 0')
    data = drop_single_value_columns(data)
    data = drop_string_columns(data)
    data = drop_correlated_features(data, target=None, threshold=0.95)

    return data


def get_desc_info(data):
    desc_name_dict, desc_val_dict = {}, {}
    cand_name_l = data.index.values.tolist()

    for idx, cand_name in enumerate(cand_name_l):
        desc_name = data.columns.values.tolist()
        desc_name_dict[cand_name] = desc_name

        desc_val = data.iloc[idx, :].values.tolist()
        desc_val_dict[cand_name] = desc_val

    return desc_name_dict, desc_val_dict


def feature_scaling(data, target=None, scaler='standard'):
    if scaler == 'minmax':
        scaler = MinMaxScaler()
    elif scaler == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError('Invalid scaling type')

    if target is not None:
        desc_df = data.drop(columns=target)
    else:
        desc_df = data.copy()

    out = scaler.fit_transform(desc_df)
    new_data = pd.DataFrame(data=out, columns=desc_df.columns)

    if target is not None:
        new_data[target] = data[target]

    return new_data


def plot_importance(importances, name):
    y = importances
    x = np.arange(len(y))

    indices = np.argsort(y)[::-1]

    plt.figure(figsize=(15, 10))
    plt.title('Feature importance (random forest regression)', fontsize=25)
    plt.bar(x, y[indices], align='center')
    plt.xticks(x, name[indices], rotation=90)
    plt.ylabel('Importance', fontsize=25)
    plt.tick_params(labelsize=20)
    plt.tight_layout()
    plt.show()


# Importance evaluation of features using random forest regression
def random_forest(results, target='barrier', visualize=False):
    X = results.drop(columns=[target]).values
    y = results.loc[:, target].values

    desc_names = results.columns.values[:-1]

    rf = RandomForestRegressor()
    rf.fit(X, y)

    desc_importances = rf.feature_importances_

    if visualize:
        plot_importance(desc_importances, desc_names)

    importance_thresh = 0.01

    isSelected = desc_importances > importance_thresh
    selected_feature_name = desc_names[isSelected]

    return selected_feature_name
