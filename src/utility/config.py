import configparser

import os


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../config.ini')
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def config_paths(resources='resources'):
    paths = load_config()['paths']
    paths['images'] = os.path.join(paths[resources], 'images')
    paths['saved_models'] = os.path.join(paths[resources], 'saved_models')
    return paths
