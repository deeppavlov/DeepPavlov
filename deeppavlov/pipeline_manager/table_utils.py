# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import List

import pandas as pd


def sort_pipes(pipes: List[dict], target_metric: str, name: str = 'results') -> List[dict]:
    """ Sorts pipelines by target metric """

    if pipes[0][name]['test']:
        sorted_logs = sorted(pipes, key=lambda x: x[name]['test'][target_metric], reverse=True)
    else:
        sorted_logs = sorted(pipes, key=lambda x: x[name]['valid'][target_metric], reverse=True)

    return sorted_logs


def get_val(v_name, def_val=None):
    return lambda x: x.get(v_name, def_val) if isinstance(x, dict) else x


def reshape_logs(logs: List[dict], max_len: int, metrics: List[str]):
    pipe_inds = []
    set_dict = {}
    for pipe_log in logs:
        pipe_inds.append(pipe_log.pop("pipe_index"))
        tmp_conf = pipe_log.pop('config')
        for i in range(max_len):
            if i < len(tmp_conf):
                pipe_log[f"component_{i + 1}"] = tmp_conf[i]
            else:
                pipe_log[f"component_{i + 1}"] = None

        tmp_res = pipe_log.pop('results')
        for data_mode, res in tmp_res.items():
            if res:
                set_dict[data_mode] = True
                for metric_name, metric_val in res.items():
                    pipe_log[f"{data_mode}_{metric_name}"] = metric_val
            else:
                set_dict[data_mode] = False
                for metric_name in metrics:
                    pipe_log[f"{data_mode}_{metric_name}"] = None

    return logs, pipe_inds, set_dict


def build_pipeline_table(log_data: list,
                         save_path: Path,
                         target_metric: str,
                         metrics_names: List[str]) -> None:
    max_pipe_len = max(len(log['config']) for log in log_data)
    log_data, pipe_inds, data_type = reshape_logs(log_data, max_pipe_len, metrics_names)

    df = pd.DataFrame(log_data, index=pipe_inds)
    df.to_csv(save_path.joinpath("exp_data.csv"))

    if data_type['test']:
        sorted_df = df.applymap(get_val('component_name')).sort_values(by=[f"test_{target_metric}"], ascending=False)
        sorted_df.to_excel(save_path.joinpath("exp_data.xlsx"))
    else:
        sorted_df = df.applymap(get_val('component_name')).sort_values(by=[f"valid_{target_metric}"], ascending=False)
        sorted_df.to_excel(save_path.joinpath("exp_data.xlsx"))
