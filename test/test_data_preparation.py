import pandas as pd
import pytest

from src.utility.data_preparation import load_data, train_val_split


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
