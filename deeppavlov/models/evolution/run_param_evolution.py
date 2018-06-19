import json
import numpy as np
import argparse
from pathlib import Path
from subprocess import Popen, PIPE
import pandas as pd
from copy import deepcopy

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
                save_path = Path(evolution.get_value_from_config(population[i],
                                                                 evolution.main_model_path + ["save_path"]))
                load_path = Path(evolution.get_value_from_config(population[i],
                                                                 evolution.main_model_path + ["load_path"]))
                population[i] = evolution.insert_value_or_dict_into_config(
                    population[i], evolution.main_model_path + ["save_path"], str(save_path.joinpath("model")))
                population[i] = evolution.insert_value_or_dict_into_config(
                    population[i], evolution.main_model_path + ["load_path"], str(load_path.joinpath("model")))

                save_path.mkdir(parents=True, exist_ok=True)
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
        with open(str(Path(evolution.get_value_from_config(
                population[i],
                evolution.main_model_path + ["save_path"])).parent.joinpath("out.txt")), "r") as fout:
            reports_data = fout.read().splitlines()[-2:]
        reports = []
        for i in range(2):
            try:
                reports.append(json.loads(reports_data[i]))
            except:
                pass
        if len(reports) == 2 and "valid" in reports[0].keys() and "test" in reports[1].keys():
            val_results = reports[0]
            test_results = reports[1]
        elif len(reports) == 1 and "valid" in reports[0].keys():
            val_results = reports[0]
        else:
            val_results = {}
            test_results = {}
            for m in CONSIDERED_METRICS:
                if "loss" in m:
                    val_results[m] = 1e6
                    test_results[m] = 1e6
                else:
                    val_results[m] = 0.
                    test_results[m] = 0.

        result_table_dict = {}
        for el in order:
            if el == "params":
                result_table_dict[el] = []
            else:
                result_table_dict[el + "_valid"] = []
                result_table_dict[el + "_test"] = []
        for m_id, m in enumerate(CONSIDERED_METRICS):
            val_metrics_path = list(evolution.find_model_path(val_results, m))[0]
            val_m = evolution.get_value_from_config(val_results, val_metrics_path + [m])
            population_metrics[m].append(val_m)
            result_table_dict[m + "_valid"].append(val_m)
            if TEST:
                test_metrics_path = list(evolution.find_model_path(test_results, m))[0]
                test_m = evolution.get_value_from_config(test_results, test_metrics_path + [m])
                result_table_dict[m + "_test"].append(test_m)
            else:
                result_table_dict[m + "_test"].append(0.)
        result_table_dict[order[-1]] = [population[i]]
        result_table = pd.DataFrame(result_table_dict)
        result_table.loc[:, result_table_columns].to_csv(result_file, index=False, sep='\t', mode='a', header=None)

    return population_metrics


parser = argparse.ArgumentParser()

parser.add_argument('--config', help='Please, enter model path to config')
parser.add_argument('--evolve_metric', help='Please, choose target metric out of given in your config.train.metrics')

parser.add_argument('--p_cross', help='Please, enter probability of crossover', type=float, default=0.2)
parser.add_argument('--pow_cross', help='Please, enter crossover power', type=float, default=0.1)
parser.add_argument('--p_mut', help='Please, enter probability of mutation', type=float, default=1.)
parser.add_argument('--pow_mut', help='Please, enter mutation power', type=float, default=0.1)

parser.add_argument('--p_size', help='Please, enter population size', type=int, default=10)
parser.add_argument('--gpus', help='Please, enter the list of visible GPUs', default="0")
parser.add_argument('--train_partition',
                    help='Please, enter partition of splitted train', default=1)
parser.add_argument('--start_from_population',
                    help='Please, enter the population number to start from. 0 means from scratch', default=0)
parser.add_argument('--path_to_population',
                    help='Please, enter the path to population to start from', default="")
parser.add_argument('--elitism_with_weights',
                    help='Please, enter whether to save elite models with weights or not', default=0)

args = parser.parse_args()

CONFIG_FILE = args.config
EVOLVE_METRIC = args.evolve_metric
POPULATION_SIZE = args.p_size
GPU_NUMBER = len(args.gpus)
gpus = [int(gpu) for gpu in args.gpus.split(",")]
TRAIN_PARTITION = int(args.train_partition)
START_FROM_POPULATION = int(args.start_from_population)
PATH_TO_POPULATION = args.path_to_population
ELITISM_WITH_WEIGHTS = int(args.elitism_with_weights)

P_CROSSOVER = args.p_cross
POW_CROSSOVER = args.pow_cross
P_MUTATION = args.p_mut
POW_MUTATION = args.pow_mut

with open(CONFIG_FILE, "r") as f:
    basic_params = json.load(f)

print("Given basic params: {}\n".format(json.dumps(basic_params, indent=2)))

evolution = ParamsEvolution(population_size=POPULATION_SIZE,
                            p_crossover=P_CROSSOVER, crossover_power=POW_CROSSOVER,
                            p_mutation=P_MUTATION, mutation_power=POW_MUTATION,
                            key_main_model="main",
                            seed=42,
                            train_partition=TRAIN_PARTITION,
                            elitism_with_weights=ELITISM_WITH_WEIGHTS,
                            **basic_params)

CONSIDERED_METRICS = evolution.get_value_from_config(evolution.basic_config,
                                                     list(evolution.find_model_path(
                                                         evolution.basic_config, "metrics"))[0] + ["metrics"])
print(CONSIDERED_METRICS)
TEST = evolution.get_value_from_config(evolution.basic_config,
                                       list(evolution.find_model_path(
                                           evolution.basic_config, "test_best"))[0] + ["test_best"])

# Result table
order = deepcopy(CONSIDERED_METRICS)
result_file = Path(evolution.get_value_from_config(evolution.basic_config,
                                                   evolution.main_model_path + ["save_path"])
                   ).joinpath("result_table.csv")
result_table_columns = []
result_table_dict = {}
for el in order:
    result_table_dict[el + "_valid"] = []
    result_table_dict[el + "_test"] = []
    result_table_columns.extend([el + "_valid", el + "_test"])

order.extend(["params"])
result_table_dict["params"] = []
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
    # _ = evolution.first_generation()
    iters = START_FROM_POPULATION
    print("\nIteration #{} starts\n".format(iters))

    population = []
    for i in range(POPULATION_SIZE):
        population.append(read_json(Path(PATH_TO_POPULATION).joinpath(
            "model_" + str(i)).joinpath("config.json")))
        population[i] = evolution.insert_value_or_dict_into_config(
            population[i], evolution.main_model_path + ["save_path"],
            str(Path(evolution.get_value_from_config(evolution.basic_config, evolution.main_model_path + ["save_path"])
                     ).joinpath("population_" + str(START_FROM_POPULATION)).joinpath("model_" + str(i))))

        population[i] = evolution.insert_value_or_dict_into_config(
            population[i], evolution.main_model_path + ["load_path"],
            str(Path(evolution.get_value_from_config(population[i], evolution.main_model_path + ["load_path"]).parent)))

    population_scores = score_population(population, POPULATION_SIZE, result_file)[EVOLVE_METRIC]
    print("Population scores: {}".format(population_scores))
    print("\nIteration #{} was done\n".format(iters))
    iters += 1

while True:
    print("\nIteration #{} starts\n".format(iters))
    population = evolution.next_generation(population, population_scores, iters)
    population_scores = score_population(population, POPULATION_SIZE, result_file)[EVOLVE_METRIC]
    print("Population scores: {}".format(population_scores))
    print("\nIteration #{} was done\n".format(iters))
    iters += 1

