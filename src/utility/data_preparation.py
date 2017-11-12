import os

import pandas as pd
from sklearn.model_selection import train_test_split
from src.utility.config import config_paths as paths


def load_data(name='train'):
    csv_path = os.path.join(paths()['resources'], '{}.csv'.format(name))
    if not os.path.isfile(csv_path):
        raise FileNotFoundError("File not found.")
    return pd.read_csv(csv_path, index_col='PassengerId')


def train_val_split(data, target='Survived'):
    if target not in data.columns:
        raise KeyError("There is no '{}' column in the data set.")
    y = data.pop(target)
    return train_test_split(data, y)
