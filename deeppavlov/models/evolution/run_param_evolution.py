import json
import numpy as np
import argparse
from pathlib import Path
from subprocess import Popen, PIPE
import pandas as pd
from copy import deepcopy, copy

from deeppavlov.models.evolution.evolution_param_generator import ParamsEvolution
from deeppavlov.core.common.file import save_json, read_json


def score_population(population, population_size, result_file):
    global evolution

    population_metrics = {}
    for m in CONSIDERED_METRICS:
        population_metrics[m] = []

    for k in range(POPULATION_SIZE // len(gpus) + 1):
        procs = []
        for j in range(len(gpus)):
            i = k * len(gpus) + j
            if i < POPULATION_SIZE:
                save_path = Path(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"])
                load_path = Path(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["load_path"])

                population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"] = \
                    str(save_path.joinpath("model"))
                population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["load_path"] = \
                    str(load_path.joinpath("model"))

                print(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"])
                try:
                    save_path.mkdir(parents=True)
                except FileExistsError:
                    pass

                f_name = save_path.joinpath("config.json")
                save_json(population[i], f_name)

                procs.append(Popen("CUDA_VISIBLE_DEVICES={} python ./models/evolution/train_phenotype.py {}"
                             " 1>{}/out.txt 2>{}/err.txt".format(gpus[j],
                                                                 str(f_name),
                                                                 str(save_path),
                                                                 str(save_path)
                                                                 ),
                                   shell=True, stdout=PIPE, stderr=PIPE))
        for j, proc in enumerate(procs):
            i = k * len(gpus) + j
            print(f'wait on {i}th proc')
            proc.wait()

    for i in range(population_size):
        try:
            val_results = np.loadtxt(fname=str(Path(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index][
                                                        "save_path"]).parent.joinpath("valid_results.txt")))
        except OSError or FileNotFoundError:
            val_results = [None for m in CONSIDERED_METRICS]
            for m_id, m in enumerate(CONSIDERED_METRICS):
                if "loss" in m:
                    val_results[m_id] = 1e6
                else:
                    val_results[m_id] = 0.
        if TEST:
            try:
                test_results = np.loadtxt(
                    fname=str(Path(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index][
                                       "save_path"]).parent.joinpath("test_results.txt")))
            except OSError or FileNotFoundError:
                test_results = [None for m in CONSIDERED_METRICS]
                for m_id, m in enumerate(CONSIDERED_METRICS):
                    if "loss" in m:
                        test_results[m_id] = 1e6
                    else:
                        test_results[m_id] = 0.

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

    return population_metrics


parser = argparse.ArgumentParser()

parser.add_argument('--config', help='Please, enter model path to config')
parser.add_argument('--evolve_metric', help='Please, choose target metric out of given in your config.train.metrics')
parser.add_argument('--p_size', help='Please, enter population size', type=int, default=10)
parser.add_argument('--gpus', help='Please, enter the list of visible GPUs', default=0)
parser.add_argument('--train_partition',
                    help='Please, enter partition of splitted train',
                    default=1)
parser.add_argument('--start_from_population',
                    help='Please, enter the population number to start from. 0 means from scratch',
                    default=0)
parser.add_argument('--path_to_population',
                    help='Please, enter the path to population to start from',
                    default="")

args = parser.parse_args()

CONFIG_FILE = args.config
EVOLVE_METRIC = args.evolve_metric
POPULATION_SIZE = args.p_size
GPU_NUMBER = len(args.gpus)
gpus = [int(gpu) for gpu in args.gpus.split(",")]
TRAIN_PARTITION = int(args.train_partition)
START_FROM_POPULATION = int(args.start_from_population)
PATH_TO_POPULATION = args.path_to_population

with open(CONFIG_FILE, "r") as f:
    basic_params = json.load(f)

print("Given basic params: {}\n".format(basic_params))

# list of names of considered metrics
CONSIDERED_METRICS = basic_params["train"]["metrics"]
TEST = basic_params["train"]["test_best"]


# EVOLUTION starts here!
evolution = ParamsEvolution(population_size=POPULATION_SIZE,
                            p_crossover=0.2, crossover_power=0.1,
                            p_mutation=1., mutation_power=0.1,
                            key_model_to_evolve="to_evolve",
                            key_basic_layers="basic_layers_params",
                            seed=42,
                            train_partition=TRAIN_PARTITION,
                            **basic_params)

# Result table
order = deepcopy(CONSIDERED_METRICS)
order.extend(["params"])
result_file = Path(basic_params["chainer"]["pipe"][
                       evolution.model_to_evolve_index]["save_path"]).joinpath("result_table.csv")
result_table_columns = []

result_table_dict = {}
for el in order:
    if el == "params":
        result_table_dict[el] = []
        result_table_columns.extend([el])
    else:
        result_table_dict[el + "_valid"] = []
        result_table_dict[el + "_test"] = []
        result_table_columns.extend([el + "_valid", el + "_test"])

result_table_columns.append("params")

if START_FROM_POPULATION == 0:
    result_table = pd.DataFrame(result_table_dict)
    result_table.loc[:, result_table_columns].to_csv(result_file, index=False, sep='\t')

    print("\nIteration #{} starts\n".format(0))
    population = evolution.first_generation()
    print(population)
    population_scores = score_population(population, POPULATION_SIZE, result_file)[EVOLVE_METRIC]

    iters = 1
else:
    # to define some clue params of evolution
    _ = evolution.first_generation()
    iters = START_FROM_POPULATION
    print("\nIteration #{} starts\n".format(iters))
    model_name = basic_params["chainer"]["pipe"][evolution.model_to_evolve_index]["model_name"]
    population = []

    for i in range(POPULATION_SIZE):
        population.append(read_json(Path(PATH_TO_POPULATION).joinpath(
            model_name + "_" + str(i)).joinpath("config.json")))
        population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"] = \
            str(Path(basic_params["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"]).joinpath(
                "population_" + str(START_FROM_POPULATION)).joinpath(model_name + "_" + str(i)))
        population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["load_path"] = \
            str(Path(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["load_path"]).parent)

    population_scores = score_population(population, POPULATION_SIZE, result_file)[EVOLVE_METRIC]
    print("Population scores: {}".format(population_scores))
    print("\nIteration #{} was done\n".format(iters))
    iters += 1

while True:
    print("\nIteration #{} starts\n".format(iters))
    population = evolution.next_generation(population, population_scores, iters)
    # print("Considered population: {}\nScoring...\n".format(population))
    population_scores = score_population(population, POPULATION_SIZE, result_file)[EVOLVE_METRIC]
    print("Population scores: {}".format(population_scores))
    print("\nIteration #{} was done\n".format(iters))
    iters += 1

