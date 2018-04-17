import pandas as pd
import json
import numpy as np

from deeppavlov.models.evolution.neuroevolution_param_generator import NetworkAndParamsEvolution

n_layers = 2
n_types = 3
population_size = 3
config_path = "../../configs/evolution/basic_intents_snips.json"

with open(config_path) as fin:
    config = json.load(fin)

evolution = NetworkAndParamsEvolution(n_layers, n_types,
                                      population_size,
                                      key_model_to_evolve="to_evolve",
                                      key_basic_layers="basic_layers_params",
                                      **config)

population = evolution.first_generation()
print(population)
population[0]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"] = population[0]["chainer"]["pipe"][
    evolution.model_to_evolve_index]["binary_mask"].tolist()
print(population)

evolution.crossover(population, p_crossover=0.9, crossover_power=0.5)
print(population)
