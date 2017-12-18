import os

from sklearn.neighbors import KNeighborsClassifier

from src.utility.config import config_paths
from src.utility.storage_utility import empty_dir, assert_dir_exists, save_model, load_model
import src.utility.storage_utility

paths = config_paths(resources='test_resources')
path_storage = os.path.join(paths['test_resources'], 'test_storage')
default_dir = paths['saved_models']
assert_dir_exists(path_storage)
src.utility.storage_utility.paths = paths


def test_empty_dir():
    test_file = os.path.join(path_storage, 'nonsense')
    with open(test_file, 'w') as f:
        f.write('nonsense')
    assert os.path.isfile(test_file)
    empty_dir(path_storage)
    assert os.listdir(path_storage) == []


def test_save_model_default_path():
    empty_dir(default_dir)
    clf = KNeighborsClassifier(n_neighbors=42)
    save_model(clf, file_name='default_42')
    model_path = os.path.join(default_dir, 'default_42.pkl')
    assert os.path.isfile(model_path)


def test_save_model_spec_path():
    empty_dir(path_storage)
    clf = KNeighborsClassifier(n_neighbors=42)
    save_model(clf, file_name='spec_42', save_dir=path_storage)
    assert os.path.isfile(os.path.join(path_storage, 'spec_42.pkl'))


def test_load_model():
    clf = load_model(file_name='default_42.pkl')
