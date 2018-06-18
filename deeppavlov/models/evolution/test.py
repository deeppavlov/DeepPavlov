import numpy as np
from deeppavlov.core.common.file import read_json
from copy import copy, deepcopy
import json


def _find_main_model_path(config, key_model, path=[]):
    """
    Find path to the main model in config which paths will be changed
    Args:
        config:
        key_model:

    Returns:
        path in config -- list of keys (strings and integers)
    """
    config_pointer = config
    # add_paths = []

    if type(config_pointer) is dict and key_model in config_pointer.keys():
        # main model is an element of chainer.pipe list
        # main model is a dictionary and has key key_main_model
        yield path
    else:
        if type(config_pointer) is dict:
            for key in list(config_pointer.keys()):
                for path_ in _find_main_model_path(config_pointer[key], key_model, path + [key]):
                    yield path_
        elif type(config_pointer) is list:
            for i in range(len(config_pointer)):
                for path_ in _find_main_model_path(config_pointer[i], key_model, path + [i]):
                    yield path_


def _insert_value_or_dict_into_config(config, path, value):
    config_copy = deepcopy(config)
    config_pointer = config_copy
    for el in path[:-1]:
        if type(config_pointer) is dict:
            config_pointer = config_pointer.setdefault(el, {})
        elif type(config_pointer) is list:
            config_pointer = config_pointer[el]
        else:
            pass
    config_pointer[path[-1]] = value
    return config_copy


def _get_value_from_config(config, path):
    config_copy = deepcopy(config)
    config_pointer = config_copy
    for el in path[:-1]:
        if type(config_pointer) is dict:
            config_pointer = config_pointer.setdefault(el, {})
        elif type(config_pointer) is list:
            config_pointer = config_pointer[el]
        else:
            pass
    return config_pointer[path[-1]]


def initialize_params_in_config(basic_config, paths):
    config = deepcopy(basic_config)

    for path_ in paths:
        param_name = path_[-1]
        value = _get_value_from_config(basic_config, path_)
        if type(value) is dict:
            if value.get("evolve_choice"):
                config = _insert_value_or_dict_into_config(config,
                                                           path_,
                                                           sample_params(
                                                               **{param_name: list(value["values"])})[param_name])
            elif value.get("evolve_range"):
                config = _insert_value_or_dict_into_config(config,
                                                           path_,
                                                           sample_params(
                                                               **{param_name: deepcopy(value)})[param_name])
            elif value.get("evolve_bool"):
                config = _insert_value_or_dict_into_config(config,
                                                           path_,
                                                           sample_params(
                                                               **{param_name: deepcopy(value)})[param_name])

    return config


def sample_params(**params):
    if not params:
        return {}
    else:
        params_copy = deepcopy(params)
    params_sample = dict()
    for param, param_val in params_copy.items():
        if isinstance(param_val, list):
            params_sample[param] = np.random.choice(param_val)
        elif isinstance(param_val, dict):
            if 'evolve_bool' in param_val and param_val['evolve_bool']:
                sample = bool(np.random.choice([True, False]))
            elif 'evolve_range' in param_val:
                sample = _sample_from_ranges(param_val)
            params_sample[param] = sample
        else:
            params_sample[param] = params_copy[param]
    return params_sample


def _sample_from_ranges(opts):
    from_ = opts['evolve_range'][0]
    to_ = opts['evolve_range'][1]
    if opts.get('scale', None) == 'log':
        sample = _sample_log(from_, to_)
    else:
        sample = np.random.uniform(from_, to_)
    if opts.get('discrete', False):
        sample = int(np.round(sample))
    return sample


def _sample_log(from_, to_):
    sample = np.exp(np.random.uniform(np.log(from_), np.log(to_)))
    return float(sample)


config = read_json("/home/dilyara/Documents/GitHub/deeppavlov_evolution/deeppavlov/configs/evolution/intents_snips.json")
paths = list(_find_main_model_path(config, "evolve_range"))

print(paths)

for t in ["evolve_range", "evolve_choice", "evolve_bool"]:
    paths = list(_find_main_model_path(config, t))
    config = initialize_params_in_config(config, paths)

print(json.dumps(config, indent=2))
