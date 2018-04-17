import numpy as np
from copy import deepcopy
from pathlib import Path
import json

from deeppavlov.models.evolution.check_binary_mask import check_and_correct_binary_mask, number_to_type_layer
from deeppavlov.core.common.file import save_json, read_json

# TODO:
# if structure of config has been changed,
# please, make sure that
# `config["chainer"]["pipe"]` is a list of models one of which is a model to be evolved,
# otherwise, in the whole class change `config["chainer"]["pipe"]` to new path

class NetworkAndParamsEvolution:
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

    def __init__(self, n_layers, n_types,
                 population_size,
                 p_crossover=0.5, crossover_power=0.5,
                 p_mutation=0.5, mutation_power=0.1,
                 key_model_to_evolve="to_evolve",
                 key_basic_layers="basic_layers_params",
                 seed=None,
                 **kwargs):
        """
        Initialize evolution with random population
        Args:
            n_layers: number of available layers of each type
            n_types: number of different types of network layers
            population_size: number of individuums per generation
            p_crossover: probability to cross over for current replacement
            crossover_power: part of EVOLVING parents parameters to exchange for offsprings
            p_mutation: probability of mutation for current replacement
            mutation_power: allowed percentage of mutation
            **kwargs: basic config with parameters
        """
        self.n_types = n_types
        self.n_layers = n_layers
        self.total_nodes = self.n_types * self.n_layers
        self.binary_mask_template = np.zeros((self.total_nodes, self.total_nodes))

        self.basic_config = deepcopy(kwargs)
        self.model_to_evolve_index = self._find_model_to_evolve_index_in_pipe(self.basic_config["chainer"]["pipe"],
                                                                              key_model_to_evolve)

        self.params = deepcopy(self.basic_config.get("chainer").get("pipe")[self.model_to_evolve_index])
        self.train_params = deepcopy(self.basic_config.get("train"))
        self.basic_layers_params = self.params.pop(key_basic_layers, None)
        self.node_types = list(self.basic_layers_params.keys())
        self.nodes = np.arange(self.total_nodes)

        print("___Basic config___: {}".format(self.basic_config))
        print("___Model to evolve index in pipe___: {}".format(self.model_to_evolve_index))
        print("___Model params___: {}".format(self.params))
        print("___Train params___: {}".format(self.train_params))
        print("___Basic layers params___: {}".format(self.basic_layers_params))

        if self.basic_layers_params is None:
            print("\n\n___PARAMS EVOLUTION is being started___")
            print("___For network evolution one has to provide config file with `basic_layers_params` key___\n\n")
        else:
            print("\n\n___NETWORK AND PARAMS EVOLUTION is being started___\n\n")

        self.population_size = population_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.mutation_power = mutation_power
        self.crossover_power = crossover_power
        self.evolving_params = []
        self.n_evolving_params = None
        self.evolving_train_params = []
        self.n_evolving_train_params = None

        if seed is None:
            pass
        else:
            np.random.seed(seed)

    def _find_model_to_evolve_index_in_pipe(self, pipe, key):
        for element_id, element in enumerate(pipe):
            if self._check_if_model_to_evolve(element, key):
                return element_id

    def _check_if_model_to_evolve(self, model, key):
        if key in model.keys():
            return True
        else:
            return False

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
        return

    def initialize_params_in_config(self, basic_params):
        params = {}
        params_for_search = {}
        evolving_params = []

        for param_name in basic_params.keys():
            if type(basic_params[param_name]) is dict:
                if basic_params[param_name].get("choice"):
                    params_for_search[param_name] = list(basic_params[param_name]["values"])
                    evolving_params.append(param_name)
                elif basic_params[param_name].get("range"):
                    params_for_search[param_name] = deepcopy(basic_params[param_name])
                    evolving_params.append(param_name)
                else:
                    # NOT evolving params
                    params[param_name] = deepcopy(basic_params[param_name])
            else:
                # NOT evolving params
                params[param_name] = deepcopy(basic_params[param_name])

        params_for_search = deepcopy(self.sample_params(**params_for_search))

        return params, params_for_search, evolving_params

    def initialize_layers_params(self):
        all_layers_params = {}

        for node_id in range(self.total_nodes):
            node_layer, node_type = number_to_type_layer(node_id, self.n_types)
            node_key = "node_{}_{}".format(node_layer, node_type)
            layers_params, layers_params_for_search, _ = self.initialize_params_in_config(
                self.basic_layers_params[self.node_types[node_type]])

            all_layers_params[node_key] = {"node_name": self.node_types[node_type],
                                            "node_type": node_type,
                                            "node_layer": node_layer,
                                            **layers_params,
                                            **layers_params_for_search
                                            }
        return all_layers_params

    def first_generation(self, iter=0):
        """
        Initialize first generation randomly according to the given constraints is self.params
        Returns:
            first generation that consists of self.population_size individuums
        """
        population = []
        for i in range(self.population_size):
            population.append(deepcopy(self.basic_config))

            # intitializing parameters for model
            params, params_for_search, evolving_params = self.initialize_params_in_config(self.params)
            self.evolving_params.extend(evolving_params)
            # initializing parameters for train
            train_params, train_params_for_search, evolving_params = self.initialize_params_in_config(self.train_params)
            self.evolving_train_params.extend(evolving_params)

            # intitializing path to save model
            if "model_name" in params_for_search.keys():
                params["save_path"] = str(Path(self.params["save_path"]).joinpath(
                    "population_" + str(iter)).joinpath(params_for_search["model_name"] + "_" + str(i)))
            else:
                params["save_path"] = str(Path(self.params["save_path"]).joinpath(
                    "population_" + str(iter)).joinpath(self.params["model_name"] + "_" + str(i)))

            layers_params = self.initialize_layers_params()

            # exchange model and layers params from basic config to sampled model params
            population[-1]["chainer"]["pipe"][self.model_to_evolve_index] = {**params,
                                                                             **params_for_search,
                                                                             **layers_params}
            # add binary_mask intialization
            print(self.sample_binary_mask())
            print(check_and_correct_binary_mask(self.nodes, self.sample_binary_mask()))

            population[-1]["chainer"]["pipe"][self.model_to_evolve_index]["binary_mask"] = \
                check_and_correct_binary_mask(self.nodes, self.sample_binary_mask())
            # exchange train params from basic config to sampled train params
            population[-1]["train"] = {**train_params,
                                       **train_params_for_search}

        self.evolving_params = list(set(self.evolving_params))
        self.evolving_train_params = list(set(self.evolving_train_params))

        self.n_evolving_params = len(self.evolving_params)
        self.n_evolving_train_params = len(self.evolving_train_params)

        return population

    def next_generation(self, generation, scores, iter,
                        p_crossover=None, crossover_power=None,
                        p_mutation=None, mutation_power=None):
        """
        Provide an operation of replacement
        Args:
            generation: current generation (set of self.population_size configs
            scores: corresponding scores that should be maximized
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

        selected_individuals = self.selection(generation, scores)
        offsprings = self.crossover(selected_individuals, p_crossover=p_crossover, crossover_power=crossover_power)
        next = self.mutation(offsprings, p_mutation=p_mutation, mutation_power=mutation_power)
        for i in range(self.population_size):
            next[i]["model_path"] = str(Path(self.params["model_path"]).joinpath(
                "population_" + str(iter)).joinpath(next[i]["model_name"] + "_" + str(i)))

        return next

    def selection(self, population, scores):
        """
        Select self.population_size individuums (with replacement) from given population.
        Probability of i-th individuum to be selected is scores_i / sum_j(scores_j)
        Args:
            population: self.population_size individuums
            scores: corresponding score that should be maximized

        Returns:
            selected self.population_size individuums with replacement
        """
        scores = np.array(scores, dtype='float')
        scores = (scores - 1.1 * min(scores) + 0.1 * max(scores)) / max(scores)
        total = np.sum(scores)
        probas_to_be_selected = scores / total
        intervals = np.array([np.sum(probas_to_be_selected[:i]) for i in range(self.population_size)])
        selected = []
        for i in range(self.population_size):
            r = np.random.random()
            individuum = population[np.where(r > intervals)[0][-1]]
            selected.append(individuum)
        return selected

    def crossover(self, population, p_crossover, crossover_power):
        """
        Recombine randomly population in pairs and cross over them with given probability.
        Cross over from two parents produces two offsprings
        each of which contains half of the parameter values from one parent and the other half from the other parent
        Args:
            population: self.population_size individuums
            p_crossover: probability to cross over for current replacement
            crossover_power: part of EVOLVING parents parameters to exchange for offsprings

        Returns:
            self.population_size offsprings
        """
        perm = np.random.permutation(self.population_size)
        offsprings = []
        for i in range(self.population_size // 2):
            parents = population[perm[2 * i]], population[perm[2 * i + 1]]
            if self.decision(p_crossover):
                params_perm = np.random.permutation(self.n_evolving_params)
                train_params_perm = np.random.permutation(self.n_evolving_train_params)
                nodes_perm = np.random.permutation(self.total_nodes)
                binary_mask_perm = np.random_permutation(self.total_nodes * self.total_nodes)

                curr_offsprings = [deepcopy(parents[0]),
                                   deepcopy(parents[1])]

                part = int(crossover_power * self.n_evolving_params)
                train_part = int(crossover_power * self.n_evolving_train_params)
                nodes_part = int(crossover_power * self.total_nodes)
                binary_mask_part = int(crossover_power * self.total_nodes * self.total_nodes)

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

                # exchange of nodes
                for j in range(self.total_nodes - nodes_part):
                    node_layer, node_type = number_to_type_layer(nodes_perm[j], self.n_types)
                    node_key = "node_{}_{}".format(node_layer, node_type)

                    curr_offsprings[0]["chainer"]["pipe"][self.model_to_evolve_index][node_key] = deepcopy(
                        parents[0]["chainer"]["pipe"][self.model_to_evolve_index][node_key])
                    curr_offsprings[1]["chainer"]["pipe"][self.model_to_evolve_index][node_key] = deepcopy(
                        parents[1]["chainer"]["pipe"][self.model_to_evolve_index][node_key])
                for j in range(self.total_nodes - nodes_part, self.total_nodes):
                    node_layer, node_type = number_to_type_layer(nodes_perm[j], self.n_types)
                    node_key = "node_{}_{}".format(node_layer, node_type)

                    curr_offsprings[0]["chainer"]["pipe"][self.model_to_evolve_index][node_key] = deepcopy(
                        parents[1]["chainer"]["pipe"][self.model_to_evolve_index][node_key])
                    curr_offsprings[1]["chainer"]["pipe"][self.model_to_evolve_index][node_key] = deepcopy(
                        parents[0]["chainer"]["pipe"][self.model_to_evolve_index][node_key])

                # exchange of binary mask elements
                for j in range(self.total_nodes * self.total_nodes - binary_mask_part):
                    node_x, node_y = binary_mask_perm[j] // self.total_nodes, binary_mask_perm[j] % self.total_nodes

                    curr_offsprings[0]["chainer"]["pipe"][self.model_to_evolve_index]["binary_mask"][node_x][node_y] =\
                        parents[0]["chainer"]["pipe"][self.model_to_evolve_index]["binary_mask"][node_x][node_y]
                    curr_offsprings[1]["chainer"]["pipe"][self.model_to_evolve_index]["binary_mask"][node_x][node_y] =\
                        parents[1]["chainer"]["pipe"][self.model_to_evolve_index]["binary_mask"][node_x][node_y]

                for j in range(self.total_nodes * self.total_nodes - binary_mask_part,
                               self.total_nodes * self.total_nodes):
                    node_x, node_y = binary_mask_perm[j] // self.total_nodes, binary_mask_perm[j] % self.total_nodes

                    curr_offsprings[0]["chainer"]["pipe"][self.model_to_evolve_index]["binary_mask"][node_x][node_y] =\
                        parents[1]["chainer"]["pipe"][self.model_to_evolve_index]["binary_mask"][node_x][node_y]
                    curr_offsprings[1]["chainer"]["pipe"][self.model_to_evolve_index]["binary_mask"][node_x][node_y] =\
                        parents[0]["chainer"]["pipe"][self.model_to_evolve_index]["binary_mask"][node_x][node_y]

                curr_offsprings[0]["chainer"]["pipe"][self.model_to_evolve_index]["binary_mask"] = \
                    check_and_correct_binary_mask(self.nodes,
                                                  curr_offsprings[0]["chainer"]["pipe"][self.model_to_evolve_index][
                                                      "binary_mask"])
                curr_offsprings[1]["chainer"]["pipe"][self.model_to_evolve_index]["binary_mask"] = \
                    check_and_correct_binary_mask(self.nodes,
                                                  curr_offsprings[1]["chainer"]["pipe"][self.model_to_evolve_index][
                                                      "binary_mask"])
                offsprings.extend(curr_offsprings)
            else:
                offsprings.extend(parents)

        if self.population_size % 2 == 1:
            offsprings.append(population[perm[-1]])
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
            mutated_individuum = {}
            for param in self.params_names:
                if self.decision(p_mutation):
                    if type(self.params[param]) is dict:
                        if self.params[param].get('discrete', False):
                            val = round(individuum[param] +
                                        ((2 * np.random.random() - 1.) * mutation_power
                                         * self.sample_params(**{param: self.params[param]})[param]))
                            val = min(max(self.params[param]["range"][0], val),
                                      self.params[param]["range"][1])
                            mutated_individuum[param] = val
                        elif 'range' in self.params[param].keys():
                            val = individuum[param] + \
                                  ((2 * np.random.random() - 1.) * mutation_power
                                   * self.sample_params(**{param: self.params[param]})[param])
                            val = min(max(self.params[param]["range"][0], val),
                                      self.params[param]["range"][1])
                            mutated_individuum[param] = val
                        elif self.params[param].get("choice"):
                            mutated_individuum[param] = individuum[param]
                    else:
                        mutated_individuum[param] = individuum[param]
                else:
                    mutated_individuum[param] = individuum[param]
            mutated.append(mutated_individuum)
        return mutated

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
            params_copy = deepcopy(self.params)
        else:
            params_copy = deepcopy(params)
        params_sample = dict()
        for param, param_val in params_copy.items():
            if isinstance(param_val, list):
                params_sample[param] = np.random.choice(param_val)
            elif isinstance(param_val, dict):
                if 'bool' in param_val and param_val['bool']:
                    sample = np.random.choice([True, False])
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

    def sample_binary_mask(self):
        return np.random.randint(0, high=2, size=self.binary_mask_template.shape).tolist()

