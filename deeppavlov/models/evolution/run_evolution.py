import json
import numpy as np
import argparse
from pathlib import Path
from subprocess import Popen, PIPE
import pandas as pd
from copy import deepcopy, copy

from deeppavlov.models.evolution.neuroevolution_param_generator import NetworkAndParamsEvolution
from deeppavlov.core.common.file import save_json


def score_population(population, population_size, result_file):
    global evolution

    population_metrics = {}
    for m in CONSIDERED_METRICS:
        population_metrics[m] = []

    procs = []

    for i in range(population_size):
        # f_name = Path(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"])
        # model_name = population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["model_name"]
        # population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"] = \
        #     str(f_name.joinpath(model_name + "_" + str(i)))
        # population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["load_path"] =\
        #     population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"]

        save_path = Path(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"])
        load_path = Path(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["load_path"])

        population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"] = \
            str(save_path.joinpath("model"))
        population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["load_path"] = \
            str(load_path.joinpath("model"))

        population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["nodes"] = \
            evolution.nodes
        print(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"])
        try:
            save_path.mkdir(parents=True)
        except FileExistsError:
            pass

        f_name = save_path.joinpath("config.json")
        population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"] =\
            population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"].tolist()
        save_json(population[i], f_name)
        procs.append(Popen("CUDA_VISIBLE_DEVICES={} python ./models/evolution/train_phenotype.py {}"
                     " 1>{}/out.txt 2>{}/err.txt".format(gpus[i],
                                                         str(f_name),
                                                         str(save_path),
                                                         str(save_path)
                                                         ),
                           shell=True, stdout=PIPE, stderr=PIPE))

    for i, proc in enumerate(procs):
        print(f'wait on {i}th proc')
        proc.wait()

    for i in range(population_size):
        val_results = np.loadtxt(fname=str(Path(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index][
                                                    "save_path"]).parent.joinpath("valid_results.txt")))
        result_table_dict = {}
        for el in order:
            result_table_dict[el] = []
        for m_id, m in enumerate(CONSIDERED_METRICS):
            result_table_dict[m].append(val_results[m_id])
        result_table_dict[order[-1]] = [population[i]]
        result_table = pd.DataFrame(result_table_dict)

        result_table.loc[:, order].to_csv(result_file, index=False, sep='\t', mode='a', header=None)

        for m_id, m in enumerate(CONSIDERED_METRICS):
            population_metrics[m].append(val_results[m_id])

        population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"] = \
            np.array(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"])

    return population_metrics


parser = argparse.ArgumentParser()

parser.add_argument('--config', help='Please, enter model path to config',
                    default='./configs/evolution/basic_intents_config.json')
parser.add_argument('--evolve_metric', help='Please, choose target metric out of given in your config.train.metrics')
parser.add_argument('--p_size', help='Please, enter population size', type=int, default=10)
parser.add_argument('--gpus', help='Please, enter the list of visible GPUs', default=0)
parser.add_argument('--n_layers', help='Please, enter number of each layer type in network', default=2)
parser.add_argument('--n_types', help='Please, enter number of types of layers', default=1)
parser.add_argument('--one_neuron_init', help='Please, enter number of types of layers', default=0)
parser.add_argument('--save_best_portion',
                    help='Please, enter portion of population to save for the next generation with weights', default=0.)
parser.add_argument('--train_partition',
                    help='Please, enter partition of splitted train', default=1)

args = parser.parse_args()

CONFIG_FILE = args.config
POPULATION_SIZE = args.p_size
GPU_NUMBER = len(args.gpus)
gpus = [int(gpu) for gpu in args.gpus.split(",")]
N_LAYERS = int(args.n_layers)
N_TYPES = int(args.n_types)
ONE_NEURON_INIT = bool(int(args.one_neuron_init))
EVOLVE_METRIC = args.evolve_metric
SAVE_BEST_PORTION = float(args.save_best_portion)
TRAIN_PARTITION = int(args.train_partition)

with open(CONFIG_FILE, "r") as f:
    basic_params = json.load(f)

print("Given basic params: {}\n".format(basic_params))

# list of names of considered metrics
CONSIDERED_METRICS = basic_params["train"]["metrics"]

# EVOLUTION starts here!
evolution = NetworkAndParamsEvolution(n_layers=N_LAYERS, n_types=N_TYPES,
                                      population_size=POPULATION_SIZE,
                                      p_crossover=0.1, crossover_power=0.5,
                                      p_mutation=0.5, mutation_power=0.1,
                                      key_model_to_evolve="to_evolve",
                                      key_basic_layers="basic_layers_params",
                                      seed=42,
                                      start_with_one_neuron=ONE_NEURON_INIT,
                                      save_best_with_weights_portion=SAVE_BEST_PORTION,
                                      train_partition=TRAIN_PARTITION,
                                      **basic_params)

# Result table
order = deepcopy(CONSIDERED_METRICS)
order.extend(["params"])
result_table_dict = {}
for el in order:
    result_table_dict[el] = []

result_file = Path(basic_params["chainer"]["pipe"][
                       evolution.model_to_evolve_index]["save_path"]).joinpath("result_table.csv")
result_table = pd.DataFrame(result_table_dict)
result_table.loc[:, order].to_csv(result_file, index=False, sep='\t')

print("\nIteration #{} starts\n".format(0))
population = evolution.first_generation()
print("Considered population: {}\nScoring...\n".format(population))
population_scores = score_population(population, POPULATION_SIZE, result_file)[EVOLVE_METRIC]

iters = 1

while True:
    print("\nIteration #{} starts\n".format(iters))

    population = evolution.next_generation(population, population_scores, iters)
    print("Considered population: {}\nScoring...\n".format(population))
    population_scores = score_population(population, POPULATION_SIZE, result_file)[EVOLVE_METRIC]
    print("Population scores: {}".format(population_scores))
    print("\nIteration #{} was done\n".format(iters))
    iters += 1

