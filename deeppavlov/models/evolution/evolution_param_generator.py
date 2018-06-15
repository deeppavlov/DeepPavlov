import numpy as np
from copy import deepcopy
from pathlib import Path
import json

from deeppavlov.models.evolution.check_binary_mask import check_and_correct_binary_mask, \
    number_to_type_layer
from deeppavlov.models.evolution.utils import find_index_of_dict_with_key_in_pipe
from deeppavlov.core.common.file import read_json


# please, make sure that
# `config["chainer"]["pipe"]` is a list of models one of which is a model to be evolved,
# otherwise, in the whole class change `config["chainer"]["pipe"]` to new path


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
                 key_model_to_evolve="to_evolve",
                 seed=None,
                 train_partition=1,
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
                            with evolving model in the basic config
            seed: random seed for initialization
            train_partition: integer number of train data parts
            **kwargs: basic config with parameters
        """

        self.basic_config = deepcopy(kwargs)
        self.model_to_evolve_index = find_index_of_dict_with_key_in_pipe(self.basic_config["chainer"]["pipe"],
                                                                         key_model_to_evolve)
        Path(self.basic_config["chainer"]["pipe"][self.model_to_evolve_index]["save_path"]).mkdir(parents=True,
                                                                                                  exist_ok=True)

        self.dataset_iterator_params = deepcopy(self.basic_config.get("dataset_iterator"))
        self.params = deepcopy(self.basic_config.get("chainer").get("pipe")[self.model_to_evolve_index])
        self.train_params = deepcopy(self.basic_config.get("train"))

        print("___Basic config___: {}".format(self.basic_config))
        print("___Model to evolve index in pipe___: {}".format(self.model_to_evolve_index))
        print("___Dataset iterator params___: {}".format(self.dataset_iterator_params))
        print("___Model params___: {}".format(self.params))
        print("___Train params___: {}".format(self.train_params))

        self.population_size = population_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.mutation_power = mutation_power
        self.crossover_power = crossover_power
        self.evolving_dataset_iterator_params = []
        self.n_evolving_dataset_iterator_params = None
        self.evolving_params = []
        self.n_evolving_params = None
        self.evolving_train_params = []
        self.n_evolving_train_params = None
        self.n_saved_best_with_weights = 0
        self.train_partition = train_partition
        self.evolution_individuum_id = 0
        self.evolution_model_id = 0

        if seed is None:
            pass
        else:
            np.random.seed(seed)

    def _insert_dict_into_model_params(self, params, model_index, dict_to_insert):
        params_copy = deepcopy(params)
        params_copy["chainer"]["pipe"].insert(model_index, dict_to_insert)
        return params_copy

    def print_dict(self, dict, string=None):
        if string is None:
            print(json.dumps(dict, indent=2))
        else:
            print(string)
            print(json.dumps(dict, indent=2))
        return None

    def initialize_params_in_config(self, basic_params):
        params = {}
        params_for_search = {}
        evolving_params = []

        for param_name in list(basic_params.keys()):
            if type(basic_params[param_name]) is dict:
                if basic_params[param_name].get("choice"):
                    params_for_search[param_name] = list(basic_params[param_name]["values"])
                    evolving_params.append(param_name)
                elif basic_params[param_name].get("range"):
                    params_for_search[param_name] = deepcopy(basic_params[param_name])
                    evolving_params.append(param_name)
                elif basic_params[param_name].get("bool"):
                    params_for_search[param_name] = deepcopy(basic_params[param_name])
                    evolving_params.append(param_name)
                else:
                    # NOT evolving params
                    params[param_name] = deepcopy(basic_params[param_name])
            else:
                # NOT evolving params
                params[param_name] = deepcopy(basic_params[param_name])
        if basic_params:
            params_for_search = deepcopy(self.sample_params(**params_for_search))

        return params, params_for_search, evolving_params

    def first_generation(self, iteration=0):
        """
        Initialize first generation randomly according to the given constraints is self.params
        Returns:
            first generation that consists of self.population_size individuums
        """
        population = []
        for i in range(self.population_size):
            population.append(deepcopy(self.basic_config))

            # initializing parameters for dataset iterator
            dataset_iterator_params, dataset_iterator_params_for_search, evolving_params = \
                self.initialize_params_in_config(self.dataset_iterator_params)
            self.evolving_dataset_iterator_params.extend(evolving_params)
            # intitializing parameters for model
            params, params_for_search, evolving_params = self.initialize_params_in_config(self.params)
            self.evolving_params.extend(evolving_params)
            # initializing parameters for train
            train_params, train_params_for_search, evolving_params = self.initialize_params_in_config(self.train_params)
            self.evolving_train_params.extend(evolving_params)

            # intitializing path to save model
            # save_path =  population_iteration/model_name_i/
            if "model_name" in params_for_search.keys():
                params["save_path"] = str(Path(self.params["save_path"]).joinpath(
                    "population_" + str(iteration)).joinpath(params_for_search["model_name"] + "_" + str(i)))
            else:
                params["save_path"] = str(Path(self.params["save_path"]).joinpath(
                    "population_" + str(iteration)).joinpath(self.params["model_name"] + "_" + str(i)))

            # load_path =  population_iteration/model_name_i/
            if "model_name" in params_for_search.keys():
                params["load_path"] = str(Path(self.params["load_path"]).joinpath(
                    "population_" + str(iteration)).joinpath(params_for_search["model_name"] + "_" + str(i)))
            else:
                params["load_path"] = str(Path(self.params["load_path"]).joinpath(
                    "population_" + str(iteration)).joinpath(self.params["model_name"] + "_" + str(i)))

            # exchange dataset iterator params from basic config to sampled train params
            population[-1]["dataset_iterator"] = {**dataset_iterator_params,
                                                  **dataset_iterator_params_for_search}
            # exchange model and layers params from basic config to sampled model params
            population[-1]["chainer"]["pipe"][self.model_to_evolve_index] = {**params,
                                                                             **params_for_search}

            # exchange train params from basic config to sampled train params
            population[-1]["train"] = {**train_params,
                                       **train_params_for_search}
            population[-1]["train"]["evolution_model_id"] = self.evolution_model_id
            self.evolution_model_id += 1

        self.evolving_dataset_iterator_params = list(set(self.evolving_dataset_iterator_params))
        self.evolving_params = list(set(self.evolving_params))
        self.evolving_train_params = list(set(self.evolving_train_params))

        self.n_evolving_dataset_iterator_params = len(self.evolving_dataset_iterator_params)
        self.n_evolving_params = len(self.evolving_params)
        self.n_evolving_train_params = len(self.evolving_train_params)

        return population

    def next_generation(self, generation, scores, iteration,
                        p_crossover=None, crossover_power=None,
                        p_mutation=None, mutation_power=None):
        """
        Provide an operation of replacement
        Args:
            generation: current generation (set of self.population_size configs
            scores: corresponding scores that should be maximized
            iteration: iteration number
            p_crossover: probability to cross over for current replacement
            crossover_power: part of parents parameters to exchange for offsprings
            p_mutation: probability of mutation for current replacement
            mutation_power: allowed percentage of mutation

        Returns:
            the next generation according to the given scores of current generation
        """
        if not p_crossover:
            p_crossover = self.p_crossover
        if not crossover_power:
            crossover_power = self.crossover_power
        if not p_mutation:
            p_mutation = self.p_mutation
        if not mutation_power:
            mutation_power = self.mutation_power

        next_population = self.selection_of_best_with_weights(generation, scores)
        print("Saved with weights: {} individuums".format(self.n_saved_best_with_weights))
        offsprings = self.crossover(generation, scores,
                                    p_crossover=p_crossover,
                                    crossover_power=crossover_power)

        changable_next = self.mutation(offsprings,
                                       p_mutation=p_mutation,
                                       mutation_power=mutation_power)

        next_population.extend(changable_next)

        for i in range(self.n_saved_best_with_weights):
            # if several train files:
            if self.train_partition != 1:
                next_population[i]["dataset_reader"]["train"] = str(Path(next_population[i]["dataset_reader"][
                                                                             "train"]).stem.split("_")[0]) \
                                                     + "_" + str(iteration % self.train_partition) + ".csv"
            try:
                # re-init learning rate with the final one
                next_population[i]["chainer"]["pipe"][self.model_to_evolve_index]["lear_rate"] = \
                    read_json(str(Path(next_population[i]["chainer"]["pipe"][self.model_to_evolve_index][
                                           "save_path"]).parent.joinpath("model_opt.json")))["final_lear_rate"]
            except:
                pass
            # paths
            next_population[i]["chainer"]["pipe"][self.model_to_evolve_index]["load_path"] = \
                str(Path(next_population[i]["chainer"]["pipe"][self.model_to_evolve_index]["save_path"]).parent)
            next_population[i]["chainer"]["pipe"][self.model_to_evolve_index]["save_path"] = \
                str(Path(self.params["save_path"]).joinpath("population_" + str(iteration)).joinpath(
                    self.params["model_name"] + "_" + str(i)))

        for i in range(self.n_saved_best_with_weights, self.population_size):
            # if several train files
            if self.train_partition != 1:
                next_population[i]["dataset_reader"]["train"] = str(Path(next_population[i]["dataset_reader"][
                                                                             "train"]).stem.split("_")[0]) \
                                                     + "_" + str(iteration % self.train_partition) + ".csv"
            # paths
            next_population[i]["chainer"]["pipe"][self.model_to_evolve_index]["save_path"] = \
                str(Path(self.params["save_path"]).joinpath("population_" + str(iteration)).joinpath(
                    self.params["model_name"] + "_" + str(i)))
            next_population[i]["chainer"]["pipe"][self.model_to_evolve_index]["load_path"] = \
                str(Path(self.params["load_path"]).joinpath("population_" + str(iteration)).joinpath(
                    self.params["model_name"] + "_" + str(i)))

            next_population[i]["train"]["evolution_model_id"] = self.evolution_model_id
            self.evolution_model_id += 1

        return next_population

    def selection_of_best_with_weights(self, population, scores):
        """
        Select individuums to save with weights for the next generation from given population.
        Range is an order of an individuum within sorted scores (1 range = max-score, self.population_size = min-score)
        Individuum with the highest score has probability equal to 1 (100%).
        Individuum with the lowest score has probability equal to 0 (0%).
        Probability of i-th individuum to be selected with weights is (a * range_i + b)
        where a = 1. / (1. - self.population_size), and
        b = self.population_size / (self.population_size - 1.)
        Args:
            population: self.population_size individuums
            scores: corresponding score that should be maximized

        Returns:
            selected self.n_saved_best_with_weights (changable) individuums
        """
        scores = np.array(scores, dtype='float')
        sorted_ids = np.argsort(scores)
        ranges = np.array([self.population_size - np.where(i == sorted_ids)[0][0]
                           for i in np.arange(self.population_size)])

        a = 1. / (1. - self.population_size)
        b = self.population_size / (self.population_size - 1.)
        probas_to_be_selected = a * ranges + b

        selected = []
        for i in range(self.population_size):
            if self.decision(probas_to_be_selected[i]):
                selected.append(deepcopy(population[i]))

        self.n_saved_best_with_weights = len(selected)
        return selected

    def crossover(self, population, scores, p_crossover, crossover_power):
        """
        Recombine randomly population in pairs and cross over them with given probability.
        Cross over from two parents produces two offsprings
        each of which contains crossover_power portion of the parameter values from one parent,
         and the other (1 - crossover_power portion) from the other parent
        Args:
            population: self.population_size individuums
            p_crossover: probability to cross over for current replacement
            crossover_power: part of EVOLVING parents parameters to exchange for offsprings

        Returns:
            (self.population_size - self.n_saved_best_with_weights) offsprings
        """
        offsprings = []
        scores = np.array(scores, dtype='float')
        probas_to_be_parent = scores / np.sum(scores)
        intervals = np.array([np.sum(probas_to_be_parent[:i]) for i in range(self.population_size)])

        for i in range(self.population_size - self.n_saved_best_with_weights):
            rs = np.random.random(2)
            parents = population[np.where(rs[0] > intervals)[0][-1]], population[np.where(rs[1] > intervals)[0][-1]]

            if self.decision(p_crossover):
                dataset_iterator_params_perm = np.random.permutation(self.n_evolving_dataset_iterator_params)
                params_perm = np.random.permutation(self.n_evolving_params)
                train_params_perm = np.random.permutation(self.n_evolving_train_params)

                curr_offsprings = [deepcopy(parents[0]),
                                   deepcopy(parents[1])]

                dataset_iterator_part = int(crossover_power * self.n_evolving_dataset_iterator_params)
                part = int(crossover_power * self.n_evolving_params)
                train_part = int(crossover_power * self.n_evolving_train_params)

                # exchange of dataset_iterator params
                for j in range(self.n_evolving_dataset_iterator_params - dataset_iterator_part):
                    curr_offsprings[0]["dataset_iterator"][
                        self.evolving_dataset_iterator_params[dataset_iterator_params_perm[j]]] = \
                        parents[0]["dataset_iterator"][
                        self.evolving_dataset_iterator_params[dataset_iterator_params_perm[j]]]
                    curr_offsprings[1]["dataset_iterator"][
                        self.evolving_dataset_iterator_params[dataset_iterator_params_perm[j]]] = \
                        parents[1]["dataset_iterator"][
                        self.evolving_dataset_iterator_params[dataset_iterator_params_perm[j]]]
                for j in range(self.n_evolving_dataset_iterator_params - dataset_iterator_part,
                               self.n_evolving_dataset_iterator_params):
                    curr_offsprings[0]["dataset_iterator"][
                        self.evolving_dataset_iterator_params[dataset_iterator_params_perm[j]]] = \
                        parents[1]["dataset_iterator"][
                        self.evolving_dataset_iterator_params[dataset_iterator_params_perm[j]]]
                    curr_offsprings[1]["dataset_iterator"][
                        self.evolving_dataset_iterator_params[dataset_iterator_params_perm[j]]] = \
                        parents[0]["dataset_iterator"][
                        self.evolving_dataset_iterator_params[dataset_iterator_params_perm[j]]]

                # exchange of model params (not layers params)
                for j in range(self.n_evolving_params - part):
                    curr_offsprings[0]["chainer"]["pipe"][self.model_to_evolve_index][
                        self.evolving_params[params_perm[j]]] = parents[0][
                        "chainer"]["pipe"][self.model_to_evolve_index][
                        self.evolving_params[params_perm[j]]]
                    curr_offsprings[1]["chainer"]["pipe"][self.model_to_evolve_index][
                        self.evolving_params[params_perm[j]]] = parents[1][
                        "chainer"]["pipe"][self.model_to_evolve_index][
                        self.evolving_params[params_perm[j]]]
                for j in range(self.n_evolving_params - part, self.n_evolving_params):
                    curr_offsprings[0]["chainer"]["pipe"][self.model_to_evolve_index][
                        self.evolving_params[params_perm[j]]] = parents[1][
                        "chainer"]["pipe"][self.model_to_evolve_index][
                        self.evolving_params[params_perm[j]]]
                    curr_offsprings[1]["chainer"]["pipe"][self.model_to_evolve_index][
                        self.evolving_params[params_perm[j]]] = parents[0][
                        "chainer"]["pipe"][self.model_to_evolve_index][
                        self.evolving_params[params_perm[j]]]

                # exchange of train params
                for j in range(self.n_evolving_train_params - train_part):
                    curr_offsprings[0]["train"][
                        self.evolving_train_params[train_params_perm[j]]] = parents[0]["train"][
                        self.evolving_train_params[train_params_perm[j]]]
                    curr_offsprings[1]["train"][
                        self.evolving_train_params[train_params_perm[j]]] = parents[1]["train"][
                        self.evolving_train_params[train_params_perm[j]]]
                for j in range(self.n_evolving_train_params - train_part, self.n_evolving_train_params):
                    curr_offsprings[0]["train"][
                        self.evolving_train_params[train_params_perm[j]]] = parents[1]["train"][
                        self.evolving_train_params[train_params_perm[j]]]
                    curr_offsprings[1]["train"][
                        self.evolving_train_params[train_params_perm[j]]] = parents[0]["train"][
                        self.evolving_train_params[train_params_perm[j]]]

                offsprings.append(deepcopy(curr_offsprings[0]))
            else:
                offsprings.append(deepcopy(parents[0]))

        return offsprings

    def mutation(self, population, p_mutation, mutation_power):
        """
        Mutate each parameter of each individuum in population with probability p_mutation
        Args:
            population: self.population_size individuums
            p_mutation: probability to mutate for each parameter
            mutation_power: allowed percentage of mutation

        Returns:
            mutated population
        """
        mutated = []

        for individuum in population:
            mutated_individuum = deepcopy(individuum)

            # mutation of dataset iterator params
            for param in self.dataset_iterator_params.keys():
                mutated_individuum["dataset_iterator"][param] = \
                    self.mutation_of_param(param, self.dataset_iterator_params,
                                           individuum["dataset_iterator"][param],
                                           p_mutation, mutation_power)

            # mutation of other model params
            for param in self.params.keys():
                mutated_individuum["chainer"]["pipe"][self.model_to_evolve_index][param] = \
                    self.mutation_of_param(param, self.params,
                                           individuum["chainer"]["pipe"][self.model_to_evolve_index][param],
                                           p_mutation, mutation_power)

            # mutation of train params
            for param in self.train_params.keys():
                mutated_individuum["train"][param] = \
                    self.mutation_of_param(param, self.train_params,
                                           individuum["train"][param],
                                           p_mutation, mutation_power)

            mutated.append(mutated_individuum)

        return mutated

    def mutation_of_param(self, param, params_dict, param_value, p_mutation, mutation_power):
        new_mutated_value = deepcopy(param_value)
        if self.decision(p_mutation):
            if type(params_dict[param]) is dict:
                if params_dict[param].get('discrete', False):
                    val = round(param_value +
                                ((2 * np.random.random() - 1.) * mutation_power
                                 * self.sample_params(**{param: params_dict[param]})[param]))
                    val = min(max(params_dict[param]["range"][0], val),
                              params_dict[param]["range"][1])
                    new_mutated_value = val
                elif 'range' in params_dict[param].keys():
                    val = param_value + \
                          ((2 * np.random.random() - 1.) * mutation_power
                           * self.sample_params(**{param: params_dict[param]})[param])
                    val = min(max(params_dict[param]["range"][0], val),
                              params_dict[param]["range"][1])
                    new_mutated_value = val
                elif params_dict[param].get("choice"):
                    # new_mutated_value = param_value
                    new_mutated_value = self.sample_params(**{param: params_dict[param]})[param]
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
            probability: probability whether

        Returns:

        """
        r = np.random.random()
        if r < probability:
            return True
        else:
            return False

    def sample_params(self, **params):
        if not params:
            return {}
        else:
            params_copy = deepcopy(params)
        params_sample = dict()
        for param, param_val in params_copy.items():
            if isinstance(param_val, list):
                params_sample[param] = np.random.choice(param_val)
            elif isinstance(param_val, dict):
                if 'bool' in param_val and param_val['bool']:
                    sample = bool(np.random.choice([True, False]))
                elif 'range' in param_val:
                    sample = self._sample_from_ranges(param_val)
                params_sample[param] = sample
            else:
                params_sample[param] = params_copy[param]
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
