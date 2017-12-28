import _pickle
import os
import shutil

from src.utility.config import config_paths

paths = config_paths()


def save_model(model, file_name, save_dir='default'):
    if save_dir is 'default':
        save_dir = paths['saved_models']
    assert_dir_exists(save_dir)
    save_file = os.path.join(save_dir, '{}.pkl'.format(file_name))
    with open(save_file, 'wb') as model_file:
        _pickle.dump(model, model_file)


def load_model(file_name, save_dir='default'):
    if save_dir is 'default':
        save_dir = paths['saved_models']
    file_path = os.path.join(save_dir, '{}.pkl'.format(file_name))
    if not os.path.isfile(file_path):
        raise FileNotFoundError
    with open(file_path, 'rb') as model_file:
        model = _pickle.load(model_file)
    return model


def assert_dir_exists(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def empty_dir(path):
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if not os.path.isdir(path):
        raise FileNotFoundError('Directory does not exist')
    inventory = os.listdir(path)
    for inv in inventory:
        full_path = os.path.join(path, inv)
        if os.path.isfile(full_path):
            os.unlink(full_path)
        elif os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            raise NameError('Path is neither linked to file nor to directory')
