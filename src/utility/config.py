import configparser

import os


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../config.ini')
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def config_paths():
    return load_config()['paths']
