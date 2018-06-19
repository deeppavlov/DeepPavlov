from copy import deepcopy
import numpy as np
import json

from deeppavlov.models.evolution.evolution_param_generator import ParamsEvolution



CONFIG_FILE = "/home/dilyara/Documents/GitHub/deeppavlov_evolution/deeppavlov/configs/evolution/intents_snips_local.json"

with open(CONFIG_FILE, "r") as f:
    basic_params = json.load(f)

# print("Given basic params: {}\n".format(json.dumps(basic_params, indent=2)))

evolution = ParamsEvolution(population_size=10,
                            **basic_params)

paths = list(evolution.find_model_path(basic_params, "evolve_range"))
print(paths)

print(evolution.get_value_from_config(basic_params, paths[0]))
