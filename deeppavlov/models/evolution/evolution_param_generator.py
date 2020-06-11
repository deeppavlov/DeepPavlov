# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import List, Any

import numpy as np

from deeppavlov.core.common.params_search import ParamsSearch
from deeppavlov.core.common.registry import register

log = getLogger(__name__)


@register('params_evolution')
class ParamsEvolution(ParamsSearch):
    """
    Class performs full evolutionary process (task scores -> max):
    1. initializes random population
    2. makes replacement to get next generation:
        a. selection according to obtained scores
        b. crossover (recombination) with given probability p_crossover
        c. mutation with given mutation rate p_mutation (probability to mutate)
            according to given mutation power sigma
            (current mutation power is randomly from -sigma to sigma)

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
        prefix: prefix to determine special keys like `PREFIX_range`, `PREFIX_bool`, `PREFIX_choice`
        **kwargs: basic config with parameters

    Attributes:
        basic_config: dictionary with initial evolutionary config
        main_model_path: list of keys and/or integers (for list) with relative path to main model (subdictionary)
        prefix: prefix to determine special keys like `PREFIX_range`, `PREFIX_bool`, `PREFIX_choice`
        population_size: number of individuums per generation
        p_crossover: probability to cross over for current replacement
        p_mutation: probability of mutation for current replacement
        mutation_power: allowed percentage of mutation
        crossover_power: part of EVOLVING parents parameters to exchange for offsprings
        elitism_with_weights: whether to save elite models with weigths or without
        n_saved_best_pretrained: number of saved models per current generation
        train_partition: integer number of train data parts
        paths_to_params: list of lists of keys and/or integers (for list)
                with relative paths to evolving parameters
        n_params: number of evolving parameters
        evolution_model_id: identity number of model (the same for loaded pre-trained models)
        models_path: path to models given in config variable `MODEL_PATH`. This variable \
            should be used as prefix to all fitted and trained model in config~
        eps: EPS value
        paths_to_fiton_dicts: list of lists of keys and/or integers (for list)\
                with relative paths to dictionaries that can be "fitted on"
        n_fiton_dicts: number of dictionaries that can be "fitted on"
        evolve_metric_optimization: whether to maximize or minimize considered metric \
                Set of Values: ``"maximize", "minimize"``
    """

    def __init__(self,
                 population_size: int,
                 p_crossover: float = 0.5, crossover_power: float = 0.5,
                 p_mutation: float = 0.5, mutation_power: float = 0.1,
                 key_main_model: str = "main",
                 seed: int = None,
                 train_partition: int = 1,
                 elitism_with_weights: bool = False,
                 prefix: str = "evolve",
                 models_path_variable: str = "MODEL_PATH",
                 **kwargs):
        """
        Initialize evolution with random population
        """
        super().__init__(prefix=prefix, seed=seed, **kwargs)

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
        self.evolution_model_id = 0
        self.basic_config, self.models_path = self.remove_key_from_config(
            self.basic_config, ["metadata", "variables", models_path_variable])
        self.models_path = Path(self.models_path)
        for path_name in ["save_path", "load_path"]:
            occured_mpaths = list(self.find_model_path(self.basic_config, path_name))
            for ppath in occured_mpaths:
                new_path = self.get_value_from_config(
                    self.basic_config,
                    ppath + [path_name]).replace(models_path_variable, "MODEL_" + path_name.upper())
                self.insert_value_or_dict_into_config(self.basic_config, ppath + [path_name], new_path)

        self.path_to_models_save_path = ["metadata", "variables", "MODEL_SAVE_PATH"]
        self.path_to_models_load_path = ["metadata", "variables", "MODEL_LOAD_PATH"]
        self.insert_value_or_dict_into_config(self.basic_config, self.path_to_models_save_path, str(self.models_path))
        self.insert_value_or_dict_into_config(self.basic_config, self.path_to_models_load_path, str(self.models_path))

        try:
            self.evolve_metric_optimization = self.get_value_from_config(
                self.basic_config, list(self.find_model_path(
                    self.basic_config, "metric_optimization"))[0] + ["metric_optimization"])
        except:
            self.evolve_metric_optimization = "maximize"

    def first_generation(self, iteration: int = 0) -> List[dict]:
        """
        Initialize first generation randomly according to the given constraints is self.params

        Args:
            iteration: number of iteration

        Returns:
            first generation that consists of self.population_size individuums
        """
        population = []
        for i in range(self.population_size):
            config = self.initialize_params_in_config(self.basic_config, self.paths_to_params)

            self.insert_value_or_dict_into_config(config, self.path_to_models_save_path,
                                                  str(self.models_path / f"population_{iteration}" / f"model_{i}"))
            self.insert_value_or_dict_into_config(config, self.path_to_models_load_path,
                                                  str(self.models_path / f"population_{iteration}" / f"model_{i}"))
            # set model_id
            config["evolution_model_id"] = self.evolution_model_id
            # next id available
            self.evolution_model_id += 1
            population.append(config)

        return population

    def next_generation(self, generation: List[dict], scores: List[float], iteration: int) -> List[dict]:
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
                    Path(next_population[i]["dataset_reader"]["train"]).stem.split("_")[:-1]
                ) + "_" + str(iteration % self.train_partition) + file_ext
            # load_paths
            if self.elitism_with_weights:
                # if elite models are saved with weights
                self.insert_value_or_dict_into_config(
                    next_population[i], self.path_to_models_load_path,
                    self.get_value_from_config(next_population[i], self.path_to_models_save_path))
            else:
                # if elite models are saved only as configurations and trained again
                self.insert_value_or_dict_into_config(
                    next_population[i], self.path_to_models_load_path,
                    str(self.models_path / f"population_{iteration}" / f"model_{i}"))

            self.insert_value_or_dict_into_config(
                next_population[i], self.path_to_models_save_path,
                str(self.models_path / f"population_{iteration}" / f"model_{i}"))

        for i in range(self.n_saved_best_pretrained, self.population_size):
            # if several train files
            if self.train_partition != 1:
                file_ext = str(Path(next_population[i]["dataset_reader"]["train"]).suffix)
                next_population[i]["dataset_reader"]["train"] = "_".join(
                    [str(p) for p in Path(next_population[i]["dataset_reader"]["train"]).stem.split("_")[:-1]]) \
                                                                + "_" + str(iteration % self.train_partition) + file_ext
            self.insert_value_or_dict_into_config(
                next_population[i], self.path_to_models_save_path,
                str(self.models_path / f"population_{iteration}" / f"model_{i}"))
            self.insert_value_or_dict_into_config(
                next_population[i], self.path_to_models_load_path,
                str(self.models_path / f"population_{iteration}" / f"model_{i}"))

            next_population[i]["evolution_model_id"] = self.evolution_model_id
            self.evolution_model_id += 1

        return next_population

    def selection_of_best_with_weights(self, population: List[dict], scores: List[float]) -> List[dict]:
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

    def range_scores(self, scores: List[float]) -> np.ndarray:
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

    def crossover(self, population: List[dict], scores: List[float]) -> List[dict]:
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
                params_perm = np.random.permutation(self.n_params)

                curr_offsprings = [deepcopy(parents[0]),
                                   deepcopy(parents[1])]

                part = int(self.crossover_power * self.n_params)

                for j in range(self.n_params - part, self.n_params):
                    self.insert_value_or_dict_into_config(curr_offsprings[0],
                                                          self.paths_to_params[
                                                              params_perm[j]],
                                                          self.get_value_from_config(
                                                              parents[1],
                                                              self.paths_to_params[
                                                                  params_perm[j]]))

                    self.insert_value_or_dict_into_config(curr_offsprings[1],
                                                          self.paths_to_params[
                                                              params_perm[j]],
                                                          self.get_value_from_config(
                                                              parents[0],
                                                              self.paths_to_params[
                                                                  params_perm[j]]))
                offsprings.append(deepcopy(curr_offsprings[0]))
            else:
                offsprings.append(deepcopy(parents[0]))

        return offsprings

    def mutation(self, population: List[dict]) -> List[dict]:
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
            for path_ in self.paths_to_params:
                param_value = self.get_value_from_config(individuum, path_)
                self.insert_value_or_dict_into_config(
                    mutated_individuum, path_,
                    self.mutation_of_param(path_, param_value))
            mutated.append(mutated_individuum)

        return mutated

    def mutation_of_param(self, param_path: list,
                          param_value: [int, float, str, list, dict, bool, np.ndarray]) -> Any:
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
            if isinstance(basic_value, dict):
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

    @staticmethod
    def decision(probability: float = 1.) -> bool:
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
