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

import json
from copy import copy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import xlsxwriter


def get_data(logs: list) -> Tuple[int, List]:
    """
    Retrieves the necessary information from the log to build a table with sorted results.

    Args:
        logs: dict; log of the experiment

    Returns:
        max_com: int; maximum pipeline length (in components)
        dataset_names: dict; dictionary with the necessary information to build a table
    """
    stop_words = ['save_path', 'load_path', 'scratch_init', 'name', 'id', 'in', 'in_y', 'out', 'fit_on']
    max_com = 0
    pipelines = []
    for val in logs:
        pipe = dict(index=int(val['pipe_index']), components=[], results={}, time=val['time'])
        # max amount of components
        if max_com < len(val['config']):
            max_com = len(val['config'])

        for component in val['config']:
            for key in stop_words:
                component.pop(key, None)
            # for key in copy(list(component.keys())):
            #     if key in stop_words:
            #         del component[key]  # dell = component.pop(key)

            comp_data = dict()
            comp_data['name'] = component.pop('component_name')
            comp_data['conf'] = component
            pipe['components'].append(comp_data)

            if 'main' in component.keys():
                break

        for name, val_ in val['results'].items():
            pipe['results'][name] = val_
        pipelines.append(pipe)

    return max_com, pipelines


def write_info(sheet: xlsxwriter, num: int, target_metric: str, cell_format: Dict, full_time: str) -> Tuple[int, int]:
    """ Write abstract info about experiment in table """
    # Start from the first cell. Rows and columns are zero indexed.
    # write info
    sheet.write(0, 0, "Number of pipelines:", cell_format)
    sheet.write(0, 1, num, cell_format)
    sheet.write(0, 2, "Target metric:", cell_format)
    sheet.write(0, 3, target_metric, cell_format)
    sheet.write(0, 4, "Experiment Time:", cell_format)
    sheet.write(0, 5, full_time, cell_format)
    sheet.write(0, 6, "(h:m:s)", cell_format)
    return 2, 0


def write_legend(sheet: xlsxwriter, row: int, col: int, data_tipe: List[str], metric_names: Union[Dict, List[str]],
                 max_com: int, cell_format: Dict) -> Tuple[int, int]:
    """ Write legend if the table """
    # write legend
    sheet.write(row, col, "Pipeline", cell_format)
    sheet.merge_range(row, col + 1, row, max_com - 1, "Preprocessing", cell_format)
    sheet.write(row, max_com, "Model", cell_format)
    for j in range(len(data_tipe)):
        p = j * len(metric_names)
        for k, met in enumerate(metric_names):
            sheet.write(row, max_com + p + k + 1, met, cell_format)
    # write pipeline run time
    sheet.write(row, max_com + len(metric_names) * len(data_tipe) + 1, "Time", cell_format)

    return row + 1, col


def write_dataset_name(sheet: xlsxwriter, sheet_2: xlsxwriter, row_1: int, row_2: int, col: int, name: str,
                       dataset_list: List[Dict], format_: Dict, max_l: int, target_metric: str,
                       metric_names: List[str]) -> None:
    """ Writes to the table the name of the dataset for which the table will be built """
    # write dataset name
    sheet.write(row_1, col, "Dataset name", format_)
    sheet.write(row_1, col + 1, name, format_)
    for l, type_d in enumerate(dataset_list[0]['results'].keys()):
        p = l * len(metric_names)
        sheet.merge_range(row_1, max_l + p + 1, row_1, max_l + p + len(metric_names), type_d, format_)
    row_1 += 1

    # write dataset name
    sheet_2.write(row_2, col, "Dataset name", format_)
    sheet_2.write(row_2, col + 1, name, format_)
    for l, type_d in enumerate(dataset_list[0]['results'].keys()):
        p = l * len(metric_names)
        sheet_2.merge_range(row_2, max_l + p + 1, row_2, max_l + p + len(metric_names), type_d, format_)
    row_2 += 1

    write_exp(row_1, row_2, col, dataset_list, sheet, sheet_2, format_, max_l, target_metric, metric_names)


def write_exp(row_1: int, row_2: int, col: int, model_list: List[Dict], sheet: xlsxwriter, sheet_2: xlsxwriter,
              _format: Dict, max_l: int, target_metric: str, metric_names: List[str]) -> None:
    """ Writes legends to the table """

    row_1, col = write_legend(sheet, row_1, col, list(model_list[0]['results'].keys()), metric_names, max_l, _format)
    row_2, col = write_legend(sheet_2, row_2, col, list(model_list[0]['results'].keys()), metric_names, max_l, _format)

    # Write pipelines table
    sorted_model_list = sort_pipes(model_list, target_metric)

    write_table(sheet, sorted_model_list, row_1, col, _format, max_l)
    # Get the best pipelines
    best_pipelines = get_best(model_list, target_metric)
    # Sorting pipelines
    best_pipelines = sort_pipes(best_pipelines, target_metric)
    # Write sort pipelines table
    write_table(sheet_2, best_pipelines, row_2, col, _format, max_l, write_conf=False)


def write_metrics(sheet: xlsxwriter, comp_dict: Dict, start_x: int, start_y: int, cell_format: Dict) -> None:
    """ Write metric to the table """
    data_names = list(comp_dict['results'].keys())
    metric_names = list(comp_dict['results'][data_names[-1]].keys())

    for j, tp in enumerate(data_names):
        p = j * len(comp_dict['results'][tp])
        for k, met in enumerate(metric_names):
            sheet.write(start_x, start_y + p + k + 1, comp_dict['results'][tp][met], cell_format)


def write_config(sheet: xlsxwriter, comp_dict: Dict, x: int, y: int, cell_format: Dict) -> None:
    """ Write config of pipeline in last cell in row"""
    z = {}
    for i, comp in enumerate(comp_dict['components']):
        z[str(i)] = comp['conf']
    s = json.dumps(z)
    sheet.write(x, y, s, cell_format)


def write_pipe(sheet: xlsxwriter, pipe_dict: Dict, start_x: int, start_y: int, cell_format: Dict, max_: int,
               write_conf: bool) -> None:
    """ Add pipeline to the table """
    data_names = list(pipe_dict['results'].keys())
    metric_names = list(pipe_dict['results'][data_names[-1]].keys())

    sheet.write(start_x, start_y, pipe_dict['index'], cell_format)
    x = start_x
    y = start_y + 1
    if len(pipe_dict['components']) > 2:
        for conf in pipe_dict['components'][:-2]:
            sheet.write(x, y, conf['name'], cell_format)
            y += 1
    if len(pipe_dict['components'][:-1]) < max_ - 1:
        sheet.merge_range(x, y, x, max_ - 1, pipe_dict['components'][-2]['name'], cell_format)
    else:
        sheet.write(x, y, pipe_dict['components'][-2]['name'], cell_format)
    sheet.write(x, max_, pipe_dict['components'][-1]['name'], cell_format)
    write_metrics(sheet, pipe_dict, x, max_, cell_format)
    # write pipeline time
    sheet.write(x, max_ + len(data_names) * len(metric_names) + 1, pipe_dict['time'], cell_format)
    # write config in second table
    if write_conf:
        write_config(sheet, pipe_dict, x, max_ + len(data_names) * len(metric_names) + 2, cell_format)


def write_table(worksheet: xlsxwriter, pipelines: Union[Dict, List[dict]], row: int, col: int, cell_format: Dict,
                max_l: int, write_conf: bool = True) -> None:
    """
    Writes a table to xlsx file.

    Args:
        worksheet: object of xlsxwriter
        pipelines: list of dicts;
        row: int; number of row
        col: int; number of column
        cell_format: dict; described format of table cell
        max_l: int; maximum len of table in cells
        write_conf: bool; determine writing config string in last cell or not

    Returns:
        row: int; number of row in table
    """
    # Write pipelines table
    for pipe in pipelines:
        write_pipe(worksheet, pipe, row, col, cell_format, max_l, write_conf)
        row += 1


def get_best(data: List[Dict], target: str) -> List[Dict]:
    """
    Calculate the best pipeline.

    Args:
        data: list of dict containing information about pipelines
        target: name of target metric

    Returns:
        best_pipes: List of pipelines
    """

    def get_name(pipeline: Dict) -> str:
        """
        Creates a short description of the pipeline as a string.

        Args:
            pipeline: pipeline config

        Returns:
            string with tiny name of pipeline
        """
        z = []
        for com in pipeline['components']:
            z.append(com['name'])
        return '->'.join(z)

    best_pipes = []
    inds = []
    buf = dict()
    for pipe in data:
        pipe_name = get_name(pipe)
        if pipe_name not in buf.keys():
            tp = list(pipe['results'].keys())[-1]
            buf[pipe_name] = {'ind': pipe['index'], 'target': pipe['results'][tp][target]}
        else:
            tp = list(pipe['results'].keys())[-1]
            if buf[pipe_name]['target'] <= pipe['results'][tp][target]:
                buf[pipe_name]['target'] = pipe['results'][tp][target]
                buf[pipe_name]['ind'] = pipe['index']

    for key, val in buf.items():
        inds.append(val['ind'])

    del buf

    for pipe in data:
        if pipe['index'] in inds:
            best_pipes.append(pipe)

    return best_pipes


def sort_pipes(pipes: List[dict], target_metric: str, name: str = 'results') -> List[dict]:
    """ Sorts pipelines by target metric """

    if 'test' in pipes[0][name].keys():
        sorted_logs = sorted(pipes, key=lambda x: x[name]['test'][target_metric], reverse=True)
    else:
        sorted_logs = sorted(pipes, key=lambda x: x[name]['valid'][target_metric], reverse=True)

    return sorted_logs


def build_pipeline_table(exp_data: dict, log_data: list, save_path: Union[str, Path]) -> None:
    """
    Creates a report table containing a brief description of all the pipelines and their results, as well as selected
    by the target metric.

    Args:
        exp_data: experiments info
        log_data: pipelines logs
        save_path: path to the folder where table will be saved

    Returns:
        None
    """
    exp_name = exp_data['exp_name']
    date = exp_data['date']
    metrics = exp_data['metrics']
    num_p = exp_data['number_of_pipes']
    target_metric = exp_data['target_metric']
    exp_time = exp_data['full_time']
    dataset_name = exp_data['dataset_name']

    # read data from log
    max_l, pipe_data = get_data(log_data)
    # create xlsx table form
    workbook = xlsxwriter.Workbook(str(save_path / 'Report_{0}_{1}.xlsx'.format(exp_name, date)))
    worksheet_1 = workbook.add_worksheet("Pipelines_sort")
    worksheet_2 = workbook.add_worksheet("Pipelines_table")
    # Create a cell format
    cell_format = workbook.add_format({'bold': 1,
                                       'border': 1,
                                       'align': 'center',
                                       'valign': 'vcenter'})
    # write legend to tables
    for wsheet in [worksheet_1, worksheet_2]:
        row, col = write_info(wsheet, num_p, target_metric, cell_format, exp_time)

    row1 = row
    row2 = row

    write_dataset_name(worksheet_2, worksheet_1, row1, row2, col, dataset_name, pipe_data, cell_format, max_l,
                       target_metric, metrics)
    workbook.close()
