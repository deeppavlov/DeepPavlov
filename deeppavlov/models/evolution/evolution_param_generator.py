import numpy as np
from copy import deepcopy
from pathlib import Path
import json
import random

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


@register('params_evolution')
class ParamsEvolution:
    """
    Class performs full evolutionary process (task scores -> max):
    1. initializes random population
    2. makes replacement to get next generation:
        a. selection according to obtained scores
        b. crossover (recombination) with given probability p_crossover
        c. mutation with given mutation rate p_mutation (probability to mutate)
            according to given mutation power sigma
            (current mutation power is randomly from -sigma to sigma)
    """

    def __init__(self,
                 population_size,
                 p_crossover=0.5, crossover_power=0.5,
                 p_mutation=0.5, mutation_power=0.1,
                 key_main_model="main",
                 seed=None,
                 train_partition=1,
                 elitism_with_weights=False,
                 **kwargs):
        """
        Initialize evolution with random population
        Args:
            population_size: number of individuums per generation
            p_crossover: probability to cross over for current replacement
            crossover_power: part of EVOLVING parents parameters to exchange for offsprings
            p_mutation: probability of mutation for current replacement
            mutation_power: allowed percentage of mutation
            key_model_to_evolve: binary flag that should be inserted into the dictionary
                        with main model in the basic config (to determine save and load paths that will be changed)
            seed: random seed for initialization
            train_partition: integer number of train data parts
            elitism_with_weights: whether to save elite models with weigths or without
            **kwargs: basic config with parameters
        """

        self.basic_config = deepcopy(kwargs)
        self.main_model_path = list(self.find_model_path(self.basic_config, key_main_model))[0]
        log.info("Main model path in config: {}".format(self.main_model_path))

        self.population_size = population_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.mutation_power = mutation_power
        self.crossover_power = crossover_power
        self.elitism_with_weights = elitism_with_weights

        self.n_saved_best_pretrained = 0
        self.train_partition = train_partition

        self.paths_to_evolving_params = []
        for evolve_type in ["evolve_range", "evolve_choice", "evolve_bool"]:
            for path_ in self.find_model_path(self.basic_config, evolve_type):
                self.paths_to_evolving_params.append(path_)

        self.n_evolving_params = len(self.paths_to_evolving_params)
        self.evolution_model_id = 0
        self.eps = 1e-6

        self.paths_to_fiton_dicts = []
        for path_ in self.find_model_path(self.basic_config, "fit_on"):
            self.paths_to_fiton_dicts.append(path_)
        self.n_fiton_dicts = len(self.paths_to_fiton_dicts)

        try:
            self.evolve_metric_optimization = self.get_value_from_config(
                self.basic_config, list(self.find_model_path(
                    self.basic_config, "metric_optimization"))[0] + ["metric_optimization"])
        except:
            self.evolve_metric_optimization = "maximize"

        if seed is None:
            pass
        else:
            np.random.seed(seed)
            random.seed(seed)

    def find_model_path(self, config, key_model, path=[]):
        """
        Find path to dictionary in config that contains key 'key_model'
        Args:
            config: dictionary
            key_model: key of sub-dictionary to be found
            path: list of keys and/or integers (for list) with relative path (needed for recursion)

        Returns:
            path in config -- list of keys (strings and integers)
        """
        config_pointer = config
        if type(config_pointer) is dict and key_model in config_pointer.keys():
            # main model is an element of chainer.pipe list
            # main model is a dictionary and has key key_main_model
            yield path
        else:
            if type(config_pointer) is dict:
                for key in list(config_pointer.keys()):
                    for path_ in self.find_model_path(config_pointer[key], key_model, path + [key]):
                        yield path_
            elif type(config_pointer) is list:
                for i in range(len(config_pointer)):
                    for path_ in self.find_model_path(config_pointer[i], key_model, path + [i]):
                        yield path_

    @staticmethod
    def insert_value_or_dict_into_config(config, path, value):
        """
        Insert value to dictionary determined by path[:-1] in field with key path[-1]
        Args:
            config: dictionary
            path: list of keys and/or integers (for list)
            value: value to be inserted

        Returns:
            config with inserted value
        """
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

    @staticmethod
    def get_value_from_config(config, path):
        """
        Return value of config element determined by path
        Args:
            config: dictionary
            path: list of keys and/or integers (for list)

        Returns:
            value
        """
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

    def initialize_params_in_config(self, basic_config, paths):
        """
        Randomly initialize all the changable parameters in config
        Args:
            basic_config: config where changable parameters are dictionaries with keys
                `evolve_range`, `evolve_bool`, `evolve_choice`
            paths: paths to changable parameters

        Returns:
            config
        """
        config = deepcopy(basic_config)
        for path_ in paths:
            param_name = path_[-1]
            value = self.get_value_from_config(basic_config, path_)
            if type(value) is dict:
                if value.get("evolve_choice") or value.get("evolve_range") or value.get("evolve_bool"):
                    config = self.insert_value_or_dict_into_config(config,
                                                                   path_,
                                                                   self.sample_params(
                                                                       **{param_name:
                                                                              deepcopy(value)})[param_name])

        return config

    def first_generation(self, iteration=0):
        """
        Initialize first generation randomly according to the given constraints is self.params
        Args:
            iteration: number of iteration

        Returns:
            first generation that consists of self.population_size individuums
        """
        population = []
        for i in range(self.population_size):
            population.append(self.initialize_params_in_config(self.basic_config, self.paths_to_evolving_params))
            for which_path in ["save_path", "load_path"]:
                population[-1] = self.insert_value_or_dict_into_config(
                    population[-1], self.main_model_path + [which_path],
                    str(Path(self.get_value_from_config(self.basic_config, self.main_model_path + [which_path])
                             ).joinpath("population_" + str(iteration)).joinpath("model_" + str(i)).joinpath("model")))
            for path_id, path_ in enumerate(self.paths_to_fiton_dicts):
                suffix = Path(self.get_value_from_config(self.basic_config,
                                                         path_ + ["save_path"])).suffix
                for which_path in ["save_path", "load_path"]:
                    population[-1] = self.insert_value_or_dict_into_config(
                        population[-1], path_ + [which_path],
                        str(Path(self.get_value_from_config(self.basic_config, self.main_model_path + [which_path])
                                 ).joinpath("population_" + str(iteration)).joinpath("model_" + str(i)).joinpath(
                            "fitted_model_" + str(path_id)).with_suffix(suffix)))
            population[-1]["evolution_model_id"] = self.evolution_model_id
            self.evolution_model_id += 1

        return population

    def next_generation(self, generation, scores, iteration):
        """
        Provide replacement
        Args:
            generation: current generation (set of self.population_size configs
            scores: corresponding scores that should be maximized
            iteration: iteration number

        Returns:
            the next generation according to the given scores of current generation
        """

        next_population = self.selection_of_best_with_weights(generation, scores)
        log.info("Saved with weights: {} models".format(self.n_saved_best_pretrained))
        offsprings = self.crossover(generation, scores)

        changable_next = self.mutation(offsprings)

        next_population.extend(changable_next)

        for i in range(self.n_saved_best_pretrained):
            # if several train files:
            if self.train_partition != 1:
                file_ext = str(Path(next_population[i]["dataset_reader"]["train"]).suffix)
                next_population[i]["dataset_reader"]["train"] = "_".join(
                    [str(p) for p in Path(next_population[i]["dataset_reader"]["train"]).stem.split("_")[:-1]])\
                                                                + "_" + str(iteration % self.train_partition) + file_ext
            try:
                # re-init learning rate with the final one (works for KerasModel)
                next_population[i] = self.insert_value_or_dict_into_config(
                    next_population[i],
                    self.main_model_path + ["lear_rate"],
                    read_json(str(Path(self.get_value_from_config(next_population[i],
                                                                  self.main_model_path + ["save_path"])
                                       ).parent.joinpath("model_opt.json")))["final_lear_rate"])
            except:
                pass

            # load_paths
            if self.elitism_with_weights:
                # if elite models are saved with weights
                next_population[i] = self.insert_value_or_dict_into_config(
                    next_population[i],
                    self.main_model_path + ["load_path"],
                    str(Path(self.get_value_from_config(next_population[i],
                                                        self.main_model_path + ["save_path"]))))
                for path_id, path_ in enumerate(self.paths_to_fiton_dicts):
                    next_population[i] = self.insert_value_or_dict_into_config(
                        next_population[i], path_ + ["load_path"],
                        str(Path(self.get_value_from_config(next_population[i],
                                                            path_ + ["save_path"]))))
            else:
                # if elite models are saved only as configurations and trained again
                next_population[i] = self.insert_value_or_dict_into_config(
                    next_population[i],
                    self.main_model_path + ["load_path"],
                    str(Path(self.get_value_from_config(self.basic_config, self.main_model_path + ["load_path"])
                             ).joinpath("population_" + str(iteration)).joinpath("model_" + str(i)).joinpath("model")))
                for path_id, path_ in enumerate(self.paths_to_fiton_dicts):
                    suffix = Path(self.get_value_from_config(self.basic_config,
                                                             path_ + ["load_path"])).suffix
                    next_population[i] = self.insert_value_or_dict_into_config(
                        next_population[i], path_ + ["load_path"],
                        str(Path(self.get_value_from_config(self.basic_config, self.main_model_path + ["load_path"])
                                 ).joinpath("population_" + str(iteration)).joinpath("model_" + str(i)).joinpath(
                            "fitted_model_" + str(path_id)).with_suffix(suffix)))

            # save_paths
            next_population[i] = self.insert_value_or_dict_into_config(
                next_population[i],
                self.main_model_path + ["save_path"],
                str(Path(self.get_value_from_config(self.basic_config, self.main_model_path + ["save_path"])
                         ).joinpath("population_" + str(iteration)).joinpath("model_" + str(i)).joinpath("model")))
            for path_id, path_ in enumerate(self.paths_to_fiton_dicts):
                suffix = Path(self.get_value_from_config(self.basic_config,
                                                         path_ + ["save_path"])).suffix
                next_population[i] = self.insert_value_or_dict_into_config(
                    next_population[i], path_ + ["save_path"],
                    str(Path(self.get_value_from_config(self.basic_config, self.main_model_path + ["save_path"])
                             ).joinpath("population_" + str(iteration)).joinpath("model_" + str(i)).joinpath(
                        "fitted_model_" + str(path_id)).with_suffix(suffix)))

        for i in range(self.n_saved_best_pretrained, self.population_size):
            # if several train files
            if self.train_partition != 1:
                file_ext = str(Path(next_population[i]["dataset_reader"]["train"]).suffix)
                next_population[i]["dataset_reader"]["train"] = "_".join(
                    [str(p) for p in Path(next_population[i]["dataset_reader"]["train"]).stem.split("_")[:-1]])\
                                                                + "_" + str(iteration % self.train_partition) + file_ext
            for which_path in ["save_path", "load_path"]:
                next_population[i] = self.insert_value_or_dict_into_config(
                    next_population[i],
                    self.main_model_path + [which_path],
                    str(Path(self.get_value_from_config(self.basic_config, self.main_model_path + [which_path])
                             ).joinpath("population_" + str(iteration)).joinpath("model_" + str(i)).joinpath("model")))
            for path_id, path_ in enumerate(self.paths_to_fiton_dicts):
                suffix = Path(self.get_value_from_config(self.basic_config,
                                                         path_ + ["save_path"])).suffix
                for which_path in ["save_path", "load_path"]:
                    next_population[i] = self.insert_value_or_dict_into_config(
                        next_population[i], path_ + [which_path],
                        str(Path(self.get_value_from_config(self.basic_config, self.main_model_path + [which_path])
                                 ).joinpath("population_" + str(iteration)).joinpath("model_" + str(i)).joinpath(
                            "fitted_model_" + str(path_id)).with_suffix(suffix)))

            next_population[i]["evolution_model_id"] = self.evolution_model_id
            self.evolution_model_id += 1

        return next_population

    def selection_of_best_with_weights(self, population, scores):
        """
        Select individuums to save with weights for the next generation from given population.
        Range is an order of an individuum within sorted scores (1 range = max-score, self.population_size = min-score)
        Individuum with the best score has probability equal to 1 (100%).
        Individuum with the worst score has probability equal to 0 (0%).
        Probability of i-th individuum to be selected with weights is (a * range_i + b)
        where a = 1. / (1. - self.population_size), and
        b = self.population_size / (self.population_size - 1.)
        Args:
            population: self.population_size individuums
            scores: list of corresponding scores

        Returns:
            selected self.n_saved_best_pretrained (changable) individuums
        """
        ranges = self.range_scores(scores)
        a = 1. / (1. - self.population_size)
        b = self.population_size / (self.population_size - 1.)
        probas_to_be_selected = a * ranges + b

        selected = []
        for i in range(self.population_size):
            if self.decision(probas_to_be_selected[i]):
                selected.append(deepcopy(population[i]))

        self.n_saved_best_pretrained = len(selected)
        return selected

    def range_scores(self, scores):
        """
        Ranges scores,
        range 1 corresponds to the best score,
        range self.population_size corresponds to the worst score.
        Args:
            scores: list of corresponding scores of population

        Returns:
            ranges
        """
        not_none_scores = np.array([x for x in scores if x is not None])
        if len(not_none_scores) == 0:
            not_none_scores = np.array([0])
        min_score = np.min(not_none_scores)
        max_score = np.max(not_none_scores)
        for i in range(self.population_size):
            if scores[i] is None:
                if self.evolve_metric_optimization == "maximize":
                    scores[i] = min_score - self.eps
                else:
                    scores[i] = max_score + self.eps
        scores = np.array(scores, dtype='float')

        sorted_ids = np.argsort(scores)
        if self.evolve_metric_optimization == "minimize":
            sorted_ids = sorted_ids[::-1]
        ranges = np.array([self.population_size - np.where(i == sorted_ids)[0][0]
                           for i in np.arange(self.population_size)])
        return ranges

    def crossover(self, population, scores):
        """
        Recombine randomly population in pairs and cross over them with given probability.
        Cross over from two parents produces two offsprings
        each of which contains crossover_power portion of the parameter values from one parent,
         and the other (1 - crossover_power portion) from the other parent
        Args:
            population: self.population_size individuums
            scores: list of corresponding scores

        Returns:
            (self.population_size - self.n_saved_best_pretained) offsprings
        """
        offsprings = []

        ranges = self.range_scores(scores)
        a = 1. / (1. - self.population_size)
        b = self.population_size / (self.population_size - 1.)
        probas_to_be_parent = (a * ranges + b) / np.sum(a * ranges + b)
        intervals = np.array([np.sum(probas_to_be_parent[:i]) for i in range(self.population_size)])

        for i in range(self.population_size - self.n_saved_best_pretrained):
            rs = np.random.random(2)
            parents = population[np.where(rs[0] > intervals)[0][-1]], population[np.where(rs[1] > intervals)[0][-1]]

            if self.decision(self.p_crossover):
                params_perm = np.random.permutation(self.n_evolving_params)

                curr_offsprings = [deepcopy(parents[0]),
                                   deepcopy(parents[1])]

                part = int(self.crossover_power * self.n_evolving_params)

                for j in range(self.n_evolving_params - part, self.n_evolving_params):
                    curr_offsprings[0] = self.insert_value_or_dict_into_config(curr_offsprings[0],
                                                                               self.paths_to_evolving_params[
                                                                                   params_perm[j]],
                                                                               self.get_value_from_config(
                                                                                   parents[1],
                                                                                   self.paths_to_evolving_params[
                                                                                       params_perm[j]]))

                    curr_offsprings[1] = self.insert_value_or_dict_into_config(curr_offsprings[1],
                                                                               self.paths_to_evolving_params[
                                                                                   params_perm[j]],
                                                                               self.get_value_from_config(
                                                                                   parents[0],
                                                                                   self.paths_to_evolving_params[
                                                                                       params_perm[j]]))
                offsprings.append(deepcopy(curr_offsprings[0]))
            else:
                offsprings.append(deepcopy(parents[0]))

        return offsprings

    def mutation(self, population):
        """
        Mutate each parameter of each individuum in population
        Args:
            population: self.population_size individuums

        Returns:
            mutated population
        """
        mutated = []

        for individuum in population:
            mutated_individuum = deepcopy(individuum)
            for path_ in self.paths_to_evolving_params:
                param_value = self.get_value_from_config(individuum, path_)
                mutated_individuum = self.insert_value_or_dict_into_config(
                    mutated_individuum, path_,
                    self.mutation_of_param(path_, param_value))
            mutated.append(mutated_individuum)

        return mutated

    def mutation_of_param(self, param_path, param_value):
        """
        Mutate particular parameter separately
        Args:
            param_path: path to parameter in basic config
            param_value: current parameter valuer

        Returns:
            mutated parameter value
        """
        if self.decision(self.p_mutation):
            param_name = param_path[-1]
            basic_value = self.get_value_from_config(self.basic_config, param_path)
            if type(basic_value) is dict:
                if basic_value.get('discrete', False):
                    val = round(param_value +
                                ((2 * np.random.random() - 1.) * self.mutation_power
                                 * self.sample_params(**{param_name: basic_value})[param_name]))
                    val = min(max(basic_value["evolve_range"][0], val),
                              basic_value["evolve_range"][1])
                    new_mutated_value = val
                elif 'evolve_range' in basic_value.keys():
                    val = param_value + \
                          ((2 * np.random.random() - 1.) * self.mutation_power
                           * self.sample_params(**{param_name: basic_value})[param_name])
                    val = min(max(basic_value["evolve_range"][0], val),
                              basic_value["evolve_range"][1])
                    new_mutated_value = val
                elif basic_value.get("evolve_choice"):
                    new_mutated_value = self.sample_params(**{param_name: basic_value})[param_name]
                elif basic_value.get("evolve_bool"):
                    new_mutated_value = self.sample_params(**{param_name: basic_value})[param_name]
                else:
                    new_mutated_value = param_value
            else:
                new_mutated_value = param_value
        else:
            new_mutated_value = param_value

        return new_mutated_value

    def decision(self, probability):
        """
        Make decision whether to do action or not with given probability
        Args:
            probability: probability whether to do action or not

        Returns:
            bool decision
        """
        r = np.random.random()
        if r < probability:
            return True
        else:
            return False

    def sample_params(self, **params):
        """
        Sample parameters according to the given possible values
        Args:
            **params: dictionary {"param_0": {"evolve_range": [0, 10]},
                                  "param_1": {"evolve_range": [0, 10], "discrete": true},
                                  "param_2": {"evolve_range": [0, 1], "scale": "log"},
                                  "param_3": {"evolve_bool": true},
                                  "param_4": [0, 1, 2, 3]}

        Returns:
            random parameter value
        """
        if not params:
            return {}
        else:
            params_copy = deepcopy(params)
        params_sample = dict()
        for param, param_val in params_copy.items():
            if isinstance(param_val, dict):
                if 'evolve_bool' in param_val and param_val['evolve_bool']:
                    sample = bool(random.choice([True, False]))
                elif 'evolve_range' in param_val:
                    sample = self._sample_from_ranges(param_val)
                elif 'evolve_choice' in param_val:
                    sample = random.choice(param_val['values'])
                params_sample[param] = sample
            else:
                params_sample[param] = params_copy[param]
        return params_sample

    def _sample_from_ranges(self, opts):
        """
        Sample parameters from ranges
        Args:
            opts: dictionary {"param_0": {"evolve_range": [0, 10]},
                              "param_1": {"evolve_range": [0, 10], "discrete": true},
                              "param_2": {"evolve_range": [0, 1], "scale": "log"}}

        Returns:
            random parameter value from range
        """
        from_ = opts['evolve_range'][0]
        to_ = opts['evolve_range'][1]
        if opts.get('scale', None) == 'log':
            sample = self._sample_log(from_, to_)
        else:
            sample = np.random.uniform(from_, to_)
        if opts.get('discrete', False):
            sample = int(np.round(sample))
        return sample

    @staticmethod
    def _sample_log(from_, to_):
        """
        Sample parameters from ranges with log scale
        Args:
            from_: lower boundary of values
            to_:  upper boundary of values

        Returns:
            random parameters value from range with log scale
        """
        sample = np.exp(np.random.uniform(np.log(from_), np.log(to_)))
        return float(sample)
