import numpy as np
from copy import deepcopy
from pathlib import Path


class Evolution:
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

    def __init__(self, population_size,
                 p_crossover=0.5, crossover_power=0.5,
                 p_mutation=0.5, mutation_power=0.1,
                 **kwargs):
        """
        Initialize evolution with random population
        Args:
            population_size: numer of individuums per generation
            p_crossover: probability to cross over for current replacement
            crossover_power: part of parents parameters to exchange for offsprings
            p_mutation: probability of mutation for current replacement
            mutation_power: allowed percentage of mutation
            **kwargs: basic config with parameters
        """
        self.params = deepcopy(kwargs)
        self.population_size = population_size
        self.p_crossover = p_crossover
        self.p_mutation = p_mutation
        self.params_names = np.array(list(self.params.keys()))
        self.n_params = len(self.params)
        self.mutation_power = mutation_power
        self.crossover_power = crossover_power

    def first_generation(self, iter=0):
        """
        Initialize first generation randomly according to the given constraints is self.params
        Returns:
            first generation that consists of self.population_size individuums
        """
        population = []
        for i in range(self.population_size):
            params = {}
            params_for_search = {}

            for param_name in self.params.keys():
                if ((type(self.params[param_name]) is str)
                        or (type(self.params[param_name]) is int)
                        or (type(self.params[param_name]) is float)
                        or (type(self.params[param_name]) is bool)
                        or (type(self.params[param_name]) is list)):
                    params[param_name] = deepcopy(self.params[param_name])
                else:
                    if self.params[param_name].get("choice"):
                        params_for_search[param_name] = list(self.params[param_name]["values"])
                    else:
                        params_for_search[param_name] = deepcopy(self.params[param_name])

            params_for_search = deepcopy(self.sample_params(**params_for_search))
            if "model_name" in params_for_search.keys():
                params["model_path"] = str(Path(self.params["model_path"]).joinpath(
                    "population_" + str(iter)).joinpath(params_for_search["model_name"] + "_" + str(i)))
            else:
                params["model_path"] = str(Path(self.params["model_path"]).joinpath(
                    "population_" + str(iter)).joinpath(self.params["model_name"] + "_" + str(i)))

            population.append({**params, **params_for_search})
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
            crossover_power: part of parents parameters to exchange for offsprings

        Returns:
            self.population_size offsprings
        """
        perm = np.random.permutation(self.population_size)
        offsprings = []
        for i in range(self.population_size // 2):
            parents = population[perm[2 * i]], population[perm[2 * i + 1]]
            if self.decision(p_crossover):
                params_perm = np.random.permutation(self.n_params)
                curr_offsprings = [{}, {}]
                part = int(crossover_power * self.n_params)
                for j in range(self.n_params - part):
                    curr_offsprings[0][self.params_names[params_perm[j]]] = parents[0][
                        self.params_names[params_perm[j]]]
                    curr_offsprings[1][self.params_names[params_perm[j]]] = parents[1][
                        self.params_names[params_perm[j]]]
                for j in range(self.n_params - part, self.n_params):
                    curr_offsprings[0][self.params_names[params_perm[j]]] = parents[1][
                        self.params_names[params_perm[j]]]
                    curr_offsprings[1][self.params_names[params_perm[j]]] = parents[0][
                        self.params_names[params_perm[j]]]
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

