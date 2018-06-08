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
        if TEST:
            test_results = np.loadtxt(fname=str(Path(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index][
                                                        "save_path"]).parent.joinpath("test_results.txt")))

        result_table_dict = {}
        for el in order:
            if el == "params":
                result_table_dict[el] = []
            else:
                result_table_dict[el + "_valid"] = []
                result_table_dict[el + "_test"] = []
        for m_id, m in enumerate(CONSIDERED_METRICS):
            result_table_dict[m + "_valid"].append(val_results[m_id])
            if TEST:
                result_table_dict[m + "_test"].append(test_results[m_id])
            else:
                result_table_dict[m + "_test"].append(0.)
        result_table_dict[order[-1]] = [population[i]]
        result_table = pd.DataFrame(result_table_dict)

        result_table.loc[:, result_table_columns].to_csv(result_file, index=False, sep='\t', mode='a', header=None)

        for m_id, m in enumerate(CONSIDERED_METRICS):
            population_metrics[m].append(val_results[m_id])

        population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"] = \
            np.array(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"])

    return population_metrics


parser = argparse.ArgumentParser()

parser.add_argument('--config', help='Please, enter model path to config')
parser.add_argument('--evolve_metric', help='Please, choose target metric out of given in your config.train.metrics')
parser.add_argument('--p_size', help='Please, enter population size', type=int, default=10)
parser.add_argument('--gpus', help='Please, enter the list of visible GPUs', default=0)
parser.add_argument('--n_layers', help='Please, enter number of each layer type in network', default=2)
parser.add_argument('--n_types', help='Please, enter number of types of layers', default=1)
parser.add_argument('--one_neuron_init', help='whether to start with zero binary mask (one neuron network)', default=0)
parser.add_argument('--given_mask_init', help='whether to start with given binary mask', default=0)
parser.add_argument('--train_partition',
                    help='Please, enter partition of splitted train',
                    default=1)

args = parser.parse_args()

CONFIG_FILE = args.config
EVOLVE_METRIC = args.evolve_metric
POPULATION_SIZE = args.p_size
GPU_NUMBER = len(args.gpus)
gpus = [int(gpu) for gpu in args.gpus.split(",")]
N_LAYERS = int(args.n_layers)
N_TYPES = int(args.n_types)
ONE_NEURON_INIT = bool(int(args.one_neuron_init))
GIVEN_MASK_INIT = bool(int(args.given_mask_init))
TRAIN_PARTITION = int(args.train_partition)

with open(CONFIG_FILE, "r") as f:
    basic_params = json.load(f)

print("Given basic params: {}\n".format(basic_params))

# list of names of considered metrics
CONSIDERED_METRICS = basic_params["train"]["metrics"]
TEST = basic_params["train"]["test_best"]

if GIVEN_MASK_INIT:
    # Embedding -> BiLSTM -> Dense -> Dense -> GlobalMaxPooling -> Dense(#classes)
    INITIAL_BINARY_MASK = np.zeros((N_TYPES * N_LAYERS, N_TYPES * N_LAYERS))
    INITIAL_BINARY_MASK[3, 0] = 1
    INITIAL_BINARY_MASK[0, N_TYPES] = 1
else:
    INITIAL_BINARY_MASK = None

# EVOLUTION starts here!
evolution = NetworkAndParamsEvolution(n_layers=N_LAYERS, n_types=N_TYPES,
                                      population_size=POPULATION_SIZE,
                                      p_crossover=0.2, crossover_power=0.1,
                                      p_mutation=1., mutation_power=0.1,
                                      key_model_to_evolve="to_evolve",
                                      key_basic_layers="basic_layers_params",
                                      seed=42,
                                      start_with_one_neuron=ONE_NEURON_INIT,
                                      train_partition=TRAIN_PARTITION,
                                      initial_binary_mask=INITIAL_BINARY_MASK,
                                      **basic_params)

# Result table
order = deepcopy(CONSIDERED_METRICS)
order.extend(["params"])

result_table_columns = []

result_table_dict = {}
for el in order:
    if order == "params":
        result_table_dict[el] = []
        result_table_columns.extend([el + "_valid"])
    else:
        result_table_dict[el + "_valid"] = []
        result_table_dict[el + "_test"] = []
        result_table_columns.extend([el + "_valid", el + "_test"])

result_table_columns.append("params")

result_file = Path(basic_params["chainer"]["pipe"][
                       evolution.model_to_evolve_index]["save_path"]).joinpath("result_table.csv")
result_table = pd.DataFrame(result_table_dict)
result_table.loc[:, result_table_columns].to_csv(result_file, index=False, sep='\t')

print("\nIteration #{} starts\n".format(0))
population = evolution.first_generation()
# print("Considered population: {}\nScoring...\n".format(population))
population_scores = score_population(population, POPULATION_SIZE, result_file)[EVOLVE_METRIC]

iters = 1

while True:
    print("\nIteration #{} starts\n".format(iters))

    population = evolution.next_generation(population, population_scores, iters)
    # print("Considered population: {}\nScoring...\n".format(population))
    population_scores = score_population(population, POPULATION_SIZE, result_file)[EVOLVE_METRIC]
    print("Population scores: {}".format(population_scores))
    print("\nIteration #{} was done\n".format(iters))
    iters += 1

