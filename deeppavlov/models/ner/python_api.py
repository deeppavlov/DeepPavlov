from typing import Union
from pathlib import Path

from deeppavlov.core.common.file import read_json
from deeppavlov.download import download_resource, get_config_downloads
from deeppavlov.deep import find_config, deep_download
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.commands.train import train_evaluate_model_from_config


CONFIGS = {'ontonotes': find_config('ner_ontonotes'),
           'collection_rus': find_config('ner_rus')}


def load_model(model_description: Union[str, dict],
               download=True):
    """ This function construct and load the model from existing config. It returns an
        instance of Chainer which is a Neural Network along with all pre-processing
        packed into a single object

    Args:
        model_description: either a string with model name or a config dictionary.
            Possible model names are: {'ontonotes', 'collection_rus'}.
        download: whether to download pre-trained model files or not

    Returns:
        model: the whole pipeline (preprocessing + Neural Network). Which is
            a callable object
    """
    config = parse_model_description(model_description)
    if download:
        download_model(config)
    model = build_model_from_config(config)
    return model


def train_model(model_description: Union[str, dict],
                data_path: Union[str, Path]):
    """ This function construct and train the model from existing config.

    Args:
        model_description: either a string with model name or a config dictionary.
            Possible model names are: {'ontonotes', 'collection_rus'}.
        data_path: path to the data in the CoNLL-2003 format on which the model
            will be trained

    Returns:
        model: the whole pipeline (preprocessing + Neural Network). Which is
            a callable object
    """
    config = parse_model_description(model_description)
    config['dataset_reader']['data_path'] = data_path or config['dataset_reader']['data_path']
    model = train_evaluate_model_from_config(config)
    return model


def parse_model_description(model_description: Union[str, dict]):
    """ Parse the model description to determine whether it is a name of the model
        or config dictionary
    """
    if isinstance(model_description, str):
        if model_description.endswith('.json'):
            return read_json(model_description)
        elif model_description not in CONFIGS:
            raise RuntimeError('The name of the model must be in {}'.format(set(CONFIGS)))
        else:
            return read_json(CONFIGS[model_description])
    elif isinstance(model_description, dict):
        return model_description
    else:
        raise RuntimeError('The model_description must be a string with model name or a config dictionary!')


def download_model(config: Union[str, dict]):
    """ Download model files given a path to the config or a config dictionary. """
    if isinstance(config, str):
        deep_download(['-c', config])
    else:
        downloads = {url: [dest] for url, dest in get_config_downloads(config)}
        for url, dest_paths in downloads.items():
            download_resource(url, dest_paths)
