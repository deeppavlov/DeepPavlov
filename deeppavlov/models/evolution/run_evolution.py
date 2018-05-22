import json
import numpy as np
import argparse
from pathlib import Path
from subprocess import Popen, PIPE
import pandas as pd

from deeppavlov.models.evolution.neuroevolution_param_generator import NetworkAndParamsEvolution

def score_population(population, population_size, result_file):
    global evolution
    population_metrics = {}
    for metric in ["classification_log_loss",
                   "classification_accuracy",
                   "classification_f1",
                   "classification_roc_auc"]:
        population_metrics[metric] = []

    procs = []

    for i in range(population_size):
        f_name = Path(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"])
        model_name = population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["model_name"]
        population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"] = \
            str(f_name.joinpath(model_name + "_" + str(i)))
        population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["load_path"] =\
            population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"]

        population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["nodes"] = \
            evolution.nodes
        print(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"])
        try:
            f_name.mkdir(parents=True)
        except FileExistsError:
            pass
        f_name = f_name.joinpath("config.json")
        population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"] =\
            population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"].tolist()
        with open(f_name, 'w') as outfile:
            json.dump(population[i], outfile)

        procs.append(Popen("CUDA_VISIBLE_DEVICES={} python ./models/evolution/train_phenotype.py {}"
                     " 1>{}/out.txt 2>{}/err.txt".format(gpus[i],
                                                         str(f_name),
                                                         str(Path(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"]).parent),
                                                         str(Path(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["save_path"]).parent)
                                                         ),
                           shell=True, stdout=PIPE, stderr=PIPE))

    for i, proc in enumerate(procs):
        print(f'wait on {i}th proc')
        proc.wait()

    for i in range(population_size):
        val_results = np.loadtxt(fname=str(Path(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index][
                                                    "save_path"]).parent.joinpath("valid_results.txt")))
        result_table = pd.DataFrame({"classification_log_loss": [val_results[0]],
                                     "classification_accuracy": [val_results[1]],
                                     "classification_f1": [val_results[2]],
                                     "classification_roc_auc": [val_results[3]],
                                     "params": [population[i]]})
        result_table.loc[:, order].to_csv(result_file, index=False, sep='\t', mode='a', header=None)
        population_metrics["classification_log_loss"].append(val_results[0])
        population_metrics["classification_accuracy"].append(val_results[1])
        population_metrics["classification_f1"].append(val_results[2])
        population_metrics["classification_roc_auc"].append(val_results[3])

        population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"] = \
            np.array(population[i]["chainer"]["pipe"][evolution.model_to_evolve_index]["binary_mask"])

    return population_metrics


parser = argparse.ArgumentParser()

parser.add_argument('--config', help='Please, enter model path to config',
                    default='./configs/evolution/basic_intents_config.json')
parser.add_argument('--p_size', help='Please, enter population size', type=int, default=10)
parser.add_argument('--gpus', help='Please, enter the list of visible GPUs', default=0)
parser.add_argument('--n_layers', help='Please, enter number of each layer type in network', default=2)
parser.add_argument('--n_types', help='Please, enter number of types of layers', default=1)
parser.add_argument('--one_neuron_init', help='Please, enter number of types of layers', default=0)
parser.add_argument('--evolve_metric', help='Please, choose target metric out of ["classification_log_loss", '
                                            '"classification_accuracy",'
                                            '      "classification_f1",'
                                            '      "classification_roc_auc"]', default="classification_roc_auc")
parser.add_argument('--save_best_portion',
                    help='Please, enter portion of population to save for the next generation with weights', default=0.)

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

with open(CONFIG_FILE, "r") as f:
    basic_params = json.load(f)

print("Given basic params: {}\n".format(basic_params))

# EVOLUTION starts here!
evolution = NetworkAndParamsEvolution(n_layers=N_LAYERS, n_types=N_TYPES,
                                      population_size=POPULATION_SIZE,
                                      p_crossover=0.1, crossover_power=0.5,
                                      p_mutation=0.5, mutation_power=0.1,
                                      key_model_to_evolve="to_evolve",
                                      key_basic_layers="basic_layers_params",
                                      seed=None,
                                      start_with_one_neuron=ONE_NEURON_INIT,
                                      save_best_with_weights_portion=SAVE_BEST_PORTION,
                                      **basic_params)

# Result table
order = ["classification_log_loss", "classification_accuracy",
         "classification_f1", "classification_roc_auc", "params"]
result_file = Path(basic_params["chainer"]["pipe"][
                       evolution.model_to_evolve_index]["save_path"]).joinpath("result_table.csv")
result_table = pd.DataFrame({"classification_log_loss": [],
                             "classification_accuracy": [],
                             "classification_f1": [],
                             "classification_roc_auc": [],
                             "params": []})
result_table.loc[:, order].to_csv(result_file, index=False, sep='\t')

print("\nIteration #{} starts\n".format(0))
population = evolution.first_generation()
print("Considered population: {}\nScoring...\n".format(population))
population_scores = score_population(population, POPULATION_SIZE, result_file)[EVOLVE_METRIC]

iters = 1

while True:
    print("\nIteration #{} starts\n".format(iters))

    population = evolution.next_generation(population, population_scores, iter=iters)
    print("Considered population: {}\nScoring...\n".format(population))
    population_scores = score_population(population, POPULATION_SIZE, result_file)[EVOLVE_METRIC]

    print("\nIteration #{} was done\n".format(iters))
    iters += 1

