import pandas as pd
import json
import numpy as np
import tensorflow as tf
from copy import deepcopy

from deeppavlov.models.evolution.neuroevolution_param_generator import NetworkAndParamsEvolution
from deeppavlov.models.evolution.evolution_intent_model import KerasEvolutionClassificationModel
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import interact_model
from deeppavlov.core.commands.utils import set_deeppavlov_root
from deeppavlov.core.common.file import save_json, read_json
from deeppavlov.models.evolution.utils import expand_tile_batch_size
from deeppavlov.models.evolution.check_binary_mask import get_digraph_from_binary_mask


n_layers = 2
n_types = 7
population_size = 1
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
population[0]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"] = population[0]["chainer"]["pipe"][
    evolution.model_to_evolve_index]["binary_mask"].tolist()

config_path = "./config_init.json"
full_config = deepcopy(population[0])
print(population[0]["chainer"]["pipe"][evolution.model_to_evolve_index])
save_json(full_config, config_path)

population[0]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"] = np.array(population[0]["chainer"]["pipe"][
    evolution.model_to_evolve_index]["binary_mask"])

population = evolution.crossover(population, p_crossover=0.9, crossover_power=0.5)
print(population)

population[0]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"] = population[0]["chainer"]["pipe"][
    evolution.model_to_evolve_index]["binary_mask"].tolist()

config_path = "./config_crossover.json"
full_config = deepcopy(population[0])
save_json(full_config, config_path)

population[0]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"] = np.array(population[0]["chainer"]["pipe"][
    evolution.model_to_evolve_index]["binary_mask"])

population = evolution.mutation(population, p_mutation=0.5, mutation_power=.5)

population[0]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"] = population[0]["chainer"]["pipe"][
    evolution.model_to_evolve_index]["binary_mask"].tolist()

config_path = "./config_mutated.json"
full_config = deepcopy(population[0])
full_config["chainer"]["pipe"][evolution.model_to_evolve_index]["nodes"] = evolution.nodes
full_config["chainer"]["pipe"][evolution.model_to_evolve_index]["total_nodes"] = evolution.total_nodes

save_json(full_config, config_path)

population[0]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"] = np.array(population[0]["chainer"]["pipe"][
    evolution.model_to_evolve_index]["binary_mask"])

dg = get_digraph_from_binary_mask(evolution.nodes,
                                  population[0]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"])

print("Edges: ", dg.edges)
train_model_from_config(config_path)



