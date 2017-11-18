import pandas as pd
import pytest

from src.utility.data_preparation import load_data, train_val_split, remove_irrelevant, one_hot_encoding


def test_wrong_data():
    with pytest.raises(FileNotFoundError):
        load_data(name='random_crap')


def test_load_data():
    global data
    data = load_data()
    assert type(data) == pd.DataFrame


def test_wrong_data_split():
    global data
    with pytest.raises(KeyError):
        train_val_split(data, target='random_crap')


def test_train_val_split():
    global data
    data_train, data_val, y_train, y_val = train_val_split(data)
    assert data_train.shape[1] == 10
    assert data_val.shape[1] == 10
    assert len(y_train.shape) == 1
    assert len(y_val.shape) == 1


def test_wrong_remove():
    global data
    with pytest.raises(KeyError):
        remove_irrelevant(data, targets=['random_crap'])


def test_remove_irrelevant():
    global data
    targets = ['Name', 'Ticket', 'Cabin']
    for target in targets:
        assert target in data.columns
    remove_irrelevant(data)
    for target in targets:
        assert target not in data.columns


def test_onehotencoding():
    global data
    test_data = data.copy()
    remove_targets = ['Embarked', 'Sex']
    targets = ['Embarked_C', 'Sex_female']

    targets_present(remove_targets, test_data)
    targets_missing(targets, test_data)

    one_hot_encoding(test_data)

    targets_missing(remove_targets, test_data)
    targets_present(targets, test_data)


def targets_missing(targets, test_data):
    for target in targets:
        assert target not in test_data.columns


def targets_present(remove_targets, test_data):
    for target in remove_targets:
        assert target in test_data.columns
