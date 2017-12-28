import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler

from src.utility.config import config_paths as paths


def load_data(name='train'):
    csv_path = os.path.join(paths()['resources'], '{}.csv'.format(name))
    if not os.path.isfile(csv_path):
        raise FileNotFoundError("File not found.")
    return pd.read_csv(csv_path, index_col='PassengerId')


def train_val_split(data, target='Survived'):
    assert_target(data, target)
    y = data.pop(target)
    return train_test_split(data, y)


def assert_target(data, target):
    if target not in data.columns:
        raise KeyError(
            "There is no '{}' column in the data set.".format(target))


def remove_irrelevant(data, targets=['Name', 'Ticket', 'Cabin']):
    for target in targets:
        assert_target(data, target)
        data.pop(target)
    return data


def one_hot_encoding(data, targets=['Sex', 'Embarked']):
    label_enc = LabelBinarizer()
    onehot_enc = OneHotEncoder()
    for target in targets:
        assert_target(data, target)
        one_hot_data = one_hot_encoded_target(data, label_enc, onehot_enc, target)
        add_onehot_classes(data, label_enc, target, one_hot_data)
        data.pop(target)


def one_hot_encoded_target(data, label_enc, onehot_enc, target):
    temp_data = label_enc.fit_transform(data[target].astype(str))
    temp_data = onehot_enc.fit_transform(temp_data)
    return temp_data


def add_onehot_classes(data, label_enc, target, temp_data):
    for i, class_ in enumerate(label_enc.classes_):
        new_class = '{}_{}'.format(target, class_)
        data[new_class] = pd.Series(data=temp_data.toarray()[:, i], index=data.index)


def split_data(data):
    data_train, data_val, y_train, y_val = train_val_split(data)
    return data_train, data_val, y_train, y_val


def load_clean_data():
    data = load_data()
    remove_irrelevant(data)
    # drop incomplete entries
    data = data.dropna()
    one_hot_encoding(data)
    return data


def standard_scale(data):
    scaler = StandardScaler()
    data = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
    return data


def load_clean_split_standard_data():
    data = load_clean_data()
    data_train, data_val, y_train, y_val = split_data(data)
    data_train = standard_scale(data_train)
    data_val = standard_scale(data_val)
    return data_train, data_val, y_train, y_val
