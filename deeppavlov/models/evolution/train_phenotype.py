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

from deeppavlov.core.commands.train import train_model_from_config, train_evaluate_model_from_config
from deeppavlov.core.common.file import read_json, save_json

config_path = sys.argv[1]

print("TRAIN PHENOTYPE")
report = train_model_from_config(config_path, is_trained=False)

# train_model_from_config(config_path)

# config = read_json(config_path)
#
# model = build_model_from_config(config, mode='infer', load_trained=True)
#
# test_model_on_data(config_path, data)
#
# val_metrics_values = np.mean(np.array(val_metrics_values), axis=0)
#
# np.savetxt(fname=Path(path_to_models).joinpath("valid_results.txt"), X=val_metrics_values)
