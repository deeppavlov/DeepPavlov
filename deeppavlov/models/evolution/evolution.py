import json
import numpy as np
import argparse
from pathlib import Path
from subprocess import Popen, PIPE
import pandas as pd


from tuning_parameters.neuroevolution_param_generator import Evolution


def score_population(population, population_size, result_file):
    population_losses = []
    population_fmeasures = []
    population_accuracies = []
    population_roc_auc_scores = []

    procs = []

    for i in range(population_size):
        f_name = Path(population[i]["model_path"])
        try:
            f_name.mkdir(parents=True)
        except FileExistsError:
            pass
        f_name = f_name.joinpath("config.json")
        with open(f_name, 'w') as outfile:
            json.dump(population[i], outfile)

        procs.append(Popen("CUDA_VISIBLE_DEVICES={} python train_phenotype.py {}"
                     " 1>{}/out.txt 2>{}/err.txt".format(gpus[i],
                                                         str(f_name),
                                                         population[i]["model_path"],
                                                         population[i]["model_path"]),
                           shell=True, stdout=PIPE, stderr=PIPE))

    for i, proc in enumerate(procs):
        print(f'wait on {i}th proc')
        proc.wait()

    for i in range(population_size):
        val_results = np.loadtxt(fname=str(Path(population[i]["model_path"]).joinpath("valid_results.txt")))
        result_table = pd.DataFrame({"loss": [val_results[0]],
                                     "accuracy": [val_results[1]],
                                     "fmeasure": [val_results[2]],
                                     "roc_auc_score": [val_results[3]],
                                     "params": [population[i]]})
        result_table.loc[:, order].to_csv(result_file, index=False, sep='\t', mode='a', header=None)
        population_losses.append(val_results[0])
        population_accuracies.append(val_results[1])
        population_fmeasures.append(val_results[2])
        population_roc_auc_scores.append(val_results[3])

    return population_roc_auc_scores


parser = argparse.ArgumentParser()

parser.add_argument('--config', help='Please, enter model path to config', default='./configs/basic_config.json')
parser.add_argument('--p_size', help='Please, enter population size', type=int, default=10)
parser.add_argument('--gpus', help='Please, enter the list of visible GPUs', default=0)

args = parser.parse_args()

CONFIG_FILE = args.config
POPULATION_SIZE = args.p_size
GPU_NUMBER = len(args.gpus)
gpus = [int(gpu) for gpu in args.gpus.split(",")]

with open(CONFIG_FILE, "r") as f:
    basic_params = json.load(f)

print("Given basic params: {}\n".format(basic_params))

try:
    Path(basic_params["model_path"]).mkdir(parents=True)
except FileExistsError:
    pass

# Result table
order = ["loss", "accuracy", "fmeasure", "roc_auc_score", "params"]
result_file = Path(basic_params["model_path"]).joinpath("result_table.csv")
result_table = pd.DataFrame({"loss": [], "accuracy": [], "fmeasure": [], "roc_auc_score": [], "params": []})
result_table.loc[:, order].to_csv(result_file, index=False, sep='\t')

# EVOLUTION starts here!
evolution = Evolution(population_size=POPULATION_SIZE, p_crossover=0.1,
                      p_mutation=0.5, mutation_power=0.1, **basic_params)

print("\nIteration #{} starts\n".format(0))
population = evolution.first_generation()
print("Considered population: {}\nScoring...\n".format(population))
population_roc_auc_scores = score_population(population, POPULATION_SIZE, result_file)

iters = 1

while True:
    print("\nIteration #{} starts\n".format(iters))

    population = evolution.next_generation(population, population_roc_auc_scores, iter=iters)
    print("Considered population: {}\nScoring...\n".format(population))
    population_roc_auc_scores = score_population(population, POPULATION_SIZE, result_file)

    print("\nIteration #{} was done\n".format(iters))
    iters += 1

