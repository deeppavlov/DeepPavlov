"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import json
import os
import sys
from logging import getLogger
from subprocess import Popen

import pandas as pd

from deeppavlov.core.commands.utils import expand_path, parse_config
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.file import read_json, save_json, find_config
from deeppavlov.models.evolution.evolution_param_generator import ParamsEvolution

log = getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument("config_path", help="path to a pipeline json config", type=str)
parser.add_argument('--key_main_model', help='key inserted in dictionary of main model in pipe', default="main")
parser.add_argument('--p_cross', help='probability of crossover', type=float, default=0.2)
parser.add_argument('--pow_cross', help='crossover power', type=float, default=0.1)
parser.add_argument('--p_mut', help='probability of mutation', type=float, default=1.)
parser.add_argument('--pow_mut', help='mutation power', type=float, default=0.1)

parser.add_argument('--p_size', help='population size', type=int, default=10)
parser.add_argument('--gpus', help='visible GPUs divided by comma <<,>>', default="-1")
parser.add_argument('--train_partition',
                    help='partition of splitted train file', default=1)
parser.add_argument('--start_from_population',
                    help='population number to start from. 0 means from scratch', default=0)
parser.add_argument('--path_to_population',
                    help='path to population to start from', default="")
parser.add_argument('--elitism_with_weights',
                    help='whether to save elite models with weights or without', action='store_true')
parser.add_argument('--iterations', help='Number of iterations', type=int, default=-1)


def main():
    args = parser.parse_args()

    pipeline_config_path = find_config(args.config_path)
    key_main_model = args.key_main_model
    population_size = args.p_size
    gpus = [int(gpu) for gpu in args.gpus.split(",")]
    train_partition = int(args.train_partition)
    start_from_population = int(args.start_from_population)
    path_to_population = args.path_to_population
    elitism_with_weights = args.elitism_with_weights
    iterations = int(args.iterations)

    p_crossover = args.p_cross
    pow_crossover = args.pow_cross
    p_mutation = args.p_mut
    pow_mutation = args.pow_mut

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        pass
    else:
        cvd = [int(gpu) for gpu in os.environ.get("CUDA_VISIBLE_DEVICES").split(",")]
        if gpus == [-1]:
            gpus = cvd
        else:
            try:
                gpus = [cvd[gpu] for gpu in gpus]
            except IndexError:
                raise ConfigError("Can not use gpus `{}` with CUDA_VISIBLE_DEVICES='{}'".format(
                    ",".join(map(str, gpus)), ",".join(map(str, cvd))
                ))

    basic_params = read_json(pipeline_config_path)
    log.info("Given basic params: {}\n".format(json.dumps(basic_params, indent=2)))

    # Initialize evolution
    evolution = ParamsEvolution(population_size=population_size,
                                p_crossover=p_crossover, crossover_power=pow_crossover,
                                p_mutation=p_mutation, mutation_power=pow_mutation,
                                key_main_model=key_main_model,
                                seed=42,
                                train_partition=train_partition,
                                elitism_with_weights=elitism_with_weights,
                                **basic_params)

    considered_metrics = evolution.get_value_from_config(evolution.basic_config,
                                                         list(evolution.find_model_path(
                                                             evolution.basic_config, "metrics"))[0] + ["metrics"])
    considered_metrics = [metric['name'] if isinstance(metric, dict) else metric for metric in considered_metrics]

    log.info(considered_metrics)
    evolve_metric = considered_metrics[0]

    # Create table variable for gathering results
    abs_path_to_main_models = expand_path(str(evolution.models_path).format(
        **evolution.basic_config['metadata']['variables']))
    abs_path_to_main_models.mkdir(parents=True, exist_ok=True)

    result_file = abs_path_to_main_models / "result_table.tsv"
    print(result_file)

    result_table_columns = []
    result_table_dict = {}
    for el in considered_metrics:
        result_table_dict[el + "_valid"] = []
        result_table_dict[el + "_test"] = []
        result_table_columns.extend([el + "_valid", el + "_test"])

    result_table_dict["params"] = []
    result_table_columns.append("params")

    if start_from_population == 0:
        # if starting evolution from scratch
        iters = 0
        result_table = pd.DataFrame(result_table_dict)
        # write down result table file
        result_table.loc[:, result_table_columns].to_csv(result_file, index=False, sep='\t')

        log.info("Iteration #{} starts".format(iters))
        # randomly generate the first population
        population = evolution.first_generation()
    else:
        # if starting evolution from already existing population
        iters = start_from_population
        log.info("Iteration #{} starts".format(iters))

        population = []
        for i in range(population_size):
            config = read_json(expand_path(path_to_population) / f"model_{i}" / "config.json")

            evolution.insert_value_or_dict_into_config(
                config, evolution.path_to_models_save_path,
                str(evolution.main_model_path / f"population_{start_from_population}" / f"model_{i}"))

            population.append(config)

    run_population(population, evolution, gpus)
    population_scores = results_to_table(population, evolution, considered_metrics,
                                         result_file, result_table_columns)[evolve_metric]
    log.info("Population scores: {}".format(population_scores))
    log.info("Iteration #{} was done".format(iters))
    iters += 1

    while True:
        if iterations != -1 and start_from_population + iterations == iters:
            log.info("End of evolution on iteration #{}".format(iters))
            break
        log.info("Iteration #{} starts".format(iters))
        population = evolution.next_generation(population, population_scores, iters)
        run_population(population, evolution, gpus)
        population_scores = results_to_table(population, evolution, considered_metrics,
                                             result_file, result_table_columns)[evolve_metric]
        log.info("Population scores: {}".format(population_scores))
        log.info("Iteration #{} was done".format(iters))
        iters += 1


def run_population(population, evolution, gpus):
    """
    Change save and load paths for obtained population, save config.json with model config,
    run population via current python executor (with which evolve.py already run)
    and on given devices (-1 means CPU, other integeres - visible for evolve.py GPUs)
    Args:
        population: list of dictionaries - configs of current population
        evolution: ParamsEvolution
        gpus: list of given devices (list of integers)

    Returns:
        None
    """
    population_size = len(population)
    for k in range(population_size // len(gpus) + 1):
        procs = []
        for j in range(len(gpus)):
            i = k * len(gpus) + j
            if i < population_size:
                save_path = expand_path(
                    evolution.get_value_from_config(parse_config(population[i]),
                                                    evolution.path_to_models_save_path))

                save_path.mkdir(parents=True, exist_ok=True)
                f_name = save_path / "config.json"
                save_json(population[i], f_name)

                with save_path.joinpath('out.txt').open('w', encoding='utf8') as outlog,\
                        save_path.joinpath('err.txt').open('w', encoding='utf8') as errlog:
                    env = dict(os.environ)
                    if len(gpus) > 1 or gpus[0] != -1:
                        env['CUDA_VISIBLE_DEVICES'] = str(gpus[j])

                    procs.append(Popen("{} -m deeppavlov train {}".format(sys.executable, str(f_name)),
                                       shell=True, stdout=outlog, stderr=errlog, env=env))
        for j, proc in enumerate(procs):
            i = k * len(gpus) + j
            log.info(f'Waiting on {i}th proc')
            if proc.wait() != 0:
                save_path = expand_path(
                    evolution.get_value_from_config(parse_config(population[i]),
                                                    evolution.path_to_models_save_path))
                with save_path.joinpath('err.txt').open(encoding='utf8') as errlog:
                    log.warning(f'Population {i} returned an error code {proc.returncode} and an error log:\n' +
                                errlog.read())
    return None


def results_to_table(population, evolution, considered_metrics, result_file, result_table_columns):
    population_size = len(population)
    validate_best = evolution.get_value_from_config(evolution.basic_config,
                                                    list(evolution.find_model_path(
                                                        evolution.basic_config, "validate_best"))[0]
                                                    + ["validate_best"])
    test_best = evolution.get_value_from_config(evolution.basic_config,
                                                list(evolution.find_model_path(
                                                    evolution.basic_config, "test_best"))[0]
                                                + ["test_best"])
    if (not validate_best) and test_best:
        log.info("Validate_best is set to False. Tuning parameters on test")
    elif (not validate_best) and (not test_best):
        raise ConfigError("Validate_best and test_best are set to False. Can not evolve.")

    population_metrics = {}
    for m in considered_metrics:
        population_metrics[m] = []
    for i in range(population_size):
        logpath = expand_path(evolution.get_value_from_config(parse_config(population[i]),
                                                              evolution.path_to_models_save_path)
                              ) / "out.txt"
        reports_data = logpath.read_text(encoding='utf8').splitlines()[-2:]
        reports = []
        for j in range(2):
            try:
                reports.append(json.loads(reports_data[j]))
            except:
                pass

        val_results = {}
        test_results = {}
        for m in considered_metrics:
            val_results[m] = None
            test_results[m] = None
        if len(reports) == 2 and "valid" in reports[0].keys() and "test" in reports[1].keys():
            val_results = reports[0]["valid"]["metrics"]
            test_results = reports[1]["test"]["metrics"]
        elif len(reports) == 2 and "valid" in reports[0].keys() and "valid" in reports[1].keys():
            val_results = reports[1]["valid"]["metrics"]
        elif len(reports) == 2 and "test" in reports[0].keys() and "test" in reports[1].keys():
            val_results = reports[1]["test"]["metrics"]
        elif len(reports) == 2 and "train" in reports[0].keys() and "valid" in reports[1].keys():
            val_results = reports[1]["valid"]["metrics"]
        elif len(reports) == 2 and "train" in reports[0].keys() and "test" in reports[1].keys():
            val_results = reports[1]["test"]["metrics"]
        elif len(reports) == 2 and "train" in reports[0].keys() and "train" in reports[1].keys():
            val_results = reports[1]["train"]["metrics"]
        elif len(reports) == 1 and "valid" in reports[0].keys():
            val_results = reports[0]["valid"]["metrics"]
        elif len(reports) == 1 and "test" in reports[0].keys():
            test_results = reports[0]["test"]["metrics"]
        else:
            raise ConfigError("Can not proceed output files: didn't find valid and/or test results")

        result_table_dict = {}
        for el in result_table_columns:
            result_table_dict[el] = []

        for m in considered_metrics:
            result_table_dict[m + "_valid"].append(val_results[m])
            result_table_dict[m + "_test"].append(test_results[m])
            if validate_best:
                population_metrics[m].append(val_results[m])
            elif test_best:
                population_metrics[m].append(test_results[m])

        result_table_dict[result_table_columns[-1]] = [json.dumps(population[i])]
        result_table = pd.DataFrame(result_table_dict)
        result_table.loc[:, result_table_columns].to_csv(result_file, index=False, sep='\t', mode='a', header=None)

    return population_metrics


if __name__ == "__main__":
    main()
