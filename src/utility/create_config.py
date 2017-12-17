import configparser

import os


def create_config():
    project_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), '../..'))
    config = configparser.ConfigParser()
    config['paths'] = {}
    config['paths']['project'] = project_path
    config['paths']['resources'] = os.path.join(project_path, 'resources')
    config['paths']['test_resources'] = os.path.join(project_path, 'test', 'test_resources')
    config_path = os.path.join(project_path, 'config.ini')
    with open(config_path, 'w') as f:
        config.write(f)


if __name__ == '__main__':
    create_config()
