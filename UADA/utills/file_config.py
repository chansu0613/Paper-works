from easydict import EasyDict
import yaml

def config(filename):
    with open(filename, 'r') as f:
        yaml_config = EasyDict(yaml.full_load(f))
    for x in yaml_config:
        print('{}: {}'.format(x, yaml_config[x]))
    return yaml_config
