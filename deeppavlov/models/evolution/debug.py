import pandas as pd
import json
import numpy as np

from deeppavlov.models.evolution.neuroevolution_param_generator import NetworkAndParamsEvolution

n_layers = 5
n_types = 7
population_size = 10
config_path = "../../configs/evolution/basic_intents_snips.json"

with open(config_path) as fin:
    config = json.load(fin)

evolution = NetworkAndParamsEvolution(n_layers, n_types,
                                      population_size,
                                      key_model_to_evolve="to_evolve",
                                      key_basic_layers="basic_layers_params",
                                      seed=42,
                                      **config)

population = evolution.first_generation()
print(population)
population[0]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"] = population[0]["chainer"]["pipe"][
    evolution.model_to_evolve_index]["binary_mask"].tolist()
print(population)

population = evolution.crossover(population, p_crossover=0.9, crossover_power=0.5)
print(population)

# print(population[0]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"])
mutated = evolution.mutation(population, p_mutation=0.5, mutation_power=.5)

for i in range(population_size):
    if (mutated[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"] !=
        population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"]).any():
        print("{} mask mutated".format(i))
