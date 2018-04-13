import numpy as np
from copy import deepcopy
from pathlib import Path


class HyperPar:
    def __init__(self, **kwargs):
        self.params = kwargs

    def sample_params(self):
        params = deepcopy(self.params)
        params_sample = dict()
        for param, param_val in params.items():
            if isinstance(param_val, list):
                params_sample[param] = np.random.choice(param_val)
            elif isinstance(param_val, dict):
                if 'bool' in param_val and param_val['bool']:
                    sample = np.random.choice([True, False])
                elif 'range' in param_val:
                    sample = self._sample_from_ranges(param_val)
                params_sample[param] = sample
            else:
                params_sample[param] = params[param]
        return params_sample

    def _sample_from_ranges(self, opts):
        from_ = opts['range'][0]
        to_ = opts['range'][1]
        if opts.get('scale', None) == 'log':
            sample = self._sample_log(from_, to_)
        else:
            sample = np.random.uniform(from_, to_)
        if opts.get('discrete', False):
            sample = int(np.round(sample))
        return sample

    @staticmethod
    def _sample_log(from_, to_):
        sample = np.exp(np.random.uniform(np.log(from_), np.log(to_)))
        return float(sample)

# net_params = HyperPar(n_filters={'range': [32, 500], 'discrete': True, 'n_samples': n_layers, 'increasing': True},
#                               filter_width={'range': [3, 11], 'discrete': True},
#                               char_embeddings_dim={'range': [10, 50], 'discrete': True},
#                               embeddings_dropout={'bool': True},
#                               dense_dropout={'bool': True},
#                               net_type=['cnn', 'rnn', 'cnn_highway'],
#                               use_crf=True,
#                               use_batch_norm=True,
#                               token_embeddings_dim=token_emb_dim,
#                               two_dense_layers=True)
# parms = net_params.sample_params()
# learning_params = HyperPar(dropout_rate={'range': [0.1, 0.9]},
# 						   epochs={'range': [10, 100], 'discrete': True},
# 						   learning_rate={'range': [1e-4, 1e-2], 'scale': 'log'},
# 						   batch_size={'range': [2, 64], 'discrete': True},
# 						   learning_rate_decay={'range': [0.3, 0.95]},
# 						   save_path='conll_models/model.ckpt').sample_params()


def get_population(basic_params, population_size, population_num):
    population = []
    for i in range(population_size):
        params = {}
        params_for_search = {}

        for param_name in basic_params.keys():
            if ((type(basic_params[param_name]) is str)
                    or (type(basic_params[param_name]) is int)
                    or (type(basic_params[param_name]) is float)
                    or (type(basic_params[param_name]) is bool)
                    or (type(basic_params[param_name]) is list)):
                params[param_name] = basic_params[param_name]
            else:
                if "values" in basic_params[param_name].keys():
                    params_for_search[param_name] = list(basic_params[param_name]["values"])
                else:
                    params_for_search[param_name] = basic_params[param_name]

        params_for_search = HyperPar(**params_for_search).sample_params()
        print()
        params["model_path"] = str(Path(basic_params["model_path"]).joinpath(
            "population_" + str(population_num)).joinpath(params_for_search["model_name"] + "_" + str(i)))
        population.append({**params, **params_for_search})
    return population
