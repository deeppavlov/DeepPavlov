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
import numpy as np
import sys
from pathlib import Path

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.common.file import read_json, save_json


config_path = sys.argv[1]

print("TRAIN PHENOTYPE")
reports = train_evaluate_model_from_config(config_path)
print(reports)

if len(reports) == 2:
    # valid and test reports
    val_metrics = dict(reports[0]["valid"]["metrics"])
    val_metrics_values = np.array(list(val_metrics.values())).reshape(-1)

    config = read_json(config_path)
    model_index = find_index_of_dict_with_key_in_pipe(pipe=config["chainer"]["pipe"],
                                                      key="to_evolve")
    np.savetxt(fname=str(Path(config["chainer"]["pipe"][model_index][
                                  "save_path"]).parent.joinpath("valid_results.txt")),
               X=val_metrics_values)

    test_metrics = dict(reports[1]["test"]["metrics"])
    test_metrics_values = np.array(list(test_metrics.values())).reshape(-1)

    config = read_json(config_path)
    model_index = find_index_of_dict_with_key_in_pipe(pipe=config["chainer"]["pipe"],
                                                      key="to_evolve")
    np.savetxt(fname=str(Path(config["chainer"]["pipe"][model_index][
                                  "save_path"]).parent.joinpath("test_results.txt")),
               X=test_metrics_values)
else:
    # valid report
    val_metrics = dict(reports[0]["valid"]["metrics"])
    val_metrics_values = np.array(list(val_metrics.values())).reshape(-1)

    config = read_json(config_path)
    model_index = find_index_of_dict_with_key_in_pipe(pipe=config["chainer"]["pipe"],
                                                      key="to_evolve")
    np.savetxt(fname=str(Path(config["chainer"]["pipe"][model_index][
                                  "save_path"]).parent.joinpath("valid_results.txt")),
               X=val_metrics_values)
