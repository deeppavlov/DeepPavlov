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

import numpy as np
import json
import xlsxwriter
import matplotlib.pyplot as plt

from os.path import join, isdir
from os import mkdir


def normal_time(z):
    if z > 1:
        h = z/3600
        m = z % 3600/60
        s = z % 3600 % 60
        t = '%i:%i:%i' % (h, m, s)
    else:
        t = '{0:.2}'.format(z)
    return t


# ------------------------------------------------Generate reports-----------------------------------------------------
# ________________________________________________Generate old table___________________________________________________
def get_pipe_sq(pipe_dict):
    names = list(pipe_dict['res'].keys())
    m = 0
    for comp in pipe_dict['components']:
        if m < len(comp['conf']):
            m = len(comp['conf'])

    height = 1 + m
    width = 1 + 2 * len(pipe_dict['components']) + len(pipe_dict['res']) * len(pipe_dict['res'][names[0]])
    return height, width


def write_component(sheet, comp_dict, start_x, start_y, mx_height, cell_format):
    delta = mx_height - len(comp_dict['conf'])
    sheet.merge_range(start_x, start_y, start_x + 1, start_y + 1, comp_dict['name'], cell_format=cell_format)

    x = start_x + 2
    for par, value in comp_dict['conf'].items():
        sheet.write(x, start_y, par, cell_format)
        sheet.write(x, start_y + 1, str(value), cell_format)
        x += 1

    for i in range(delta):
        sheet.write(x, start_y, '-', cell_format)
        sheet.write(x, start_y + 1, '-', cell_format)
        x += 1

    return None


def write_old_metrics(sheet, comp_dict, start_x, start_y, mx_height, cell_format):
    data_names = list(comp_dict['res'].keys())
    metric_names = list(comp_dict['res'][data_names[-1]].keys())

    for j, tp in enumerate(data_names):
        sheet.merge_range(start_x, start_y + j * len(metric_names), start_x + 1,
                          start_y + j * len(metric_names) + len(metric_names) - 1, tp, cell_format=cell_format)
        for k, met in enumerate(metric_names):
            sheet.write(start_x + 2, start_y + j * len(metric_names) + k, met, cell_format)

            sheet.merge_range(start_x + 3, start_y + j * len(comp_dict['res']) + k, start_x + mx_height,
                              start_y + j * len(comp_dict['res']) + k, comp_dict['res'][tp][met],
                              cell_format=cell_format)
    return None


def write_old_pipe(sheet, pipe_dict, start_x, start_y, cell_format, max_):
    height, width = get_pipe_sq(pipe_dict)
    # height = height - 1

    sheet.merge_range(start_x, start_y, start_x + 1, start_y, 'pipeline_index', cell_format)
    sheet.merge_range(start_x + 2, start_y, start_x + height, start_y, pipe_dict['index'],
                      cell_format=cell_format)

    x = start_x
    y = start_y + 1
    for conf in pipe_dict['components']:
        write_component(sheet, conf, x, y, height - 1, cell_format)
        y += 2

    dy = 2*(max_ - len(pipe_dict['components']))  # + 2
    write_old_metrics(sheet, pipe_dict, x, y + dy, height, cell_format)

    return height


def sort_old_pipes(pipes, target_metric):
    ind_val = []
    sort_pipes_ = []
    dtype = [('value', 'float'), ('index', 'int')]
    for pipe in pipes:
        if 'test' not in pipe['res'].keys():
            name = list(pipe['res'].keys())[0]
        else:
            name = 'test'
        rm = pipe['res'][name]
        ind_val.append((rm[target_metric], pipe['index']))

    ind_val = np.sort(np.array(ind_val, dtype=dtype), order='value')
    for com in ind_val:
        ind = com[1]
        for pipe in pipes:
            if pipe['index'] == ind:
                sort_pipes_.append(pipe)

    del pipes, ind_val

    sort_pipes_.reverse()

    return sort_pipes_


def build_report(log, target_metric=None, save_path='./'):
    if isinstance(log, str):
        with open(log, 'r') as lgd:
            log_data = json.load(lgd)
            lgd.close()
    elif isinstance(log, dict):
        log_data = log
    else:
        raise ValueError("Input must be a strings (path to the json logfile) or a dict with log data,"
                         " but {} was found.".format(type(log)))

    exp_name = log_data['experiment_info']['exp_name']
    date = log_data['experiment_info']['date']
    metrics = log_data['experiment_info']['metrics']
    if target_metric is None:
        target_metric = metrics[0]

    workbook = xlsxwriter.Workbook(join(save_path, 'Old_report_{0}_{1}.xlsx'.format(exp_name, date)))
    worksheet = workbook.add_worksheet("Pipelines_table")

    pipelines = []
    max_com = 0
    for model_name, val in log_data['experiments'].items():
        for num, conf in val.items():
            pipe = dict(index=int(num), components=[], res={})
            # max amount of components
            if max_com < len(conf['config']):
                max_com = len(conf['config'])

            for component in conf['config']:
                comp_data = {}
                comp_data['name'] = component.pop('component_name')

                if 'save_path' in component.keys():
                    del component['save_path']
                if 'load_path' in component.keys():
                    del component['load_path']
                if 'scratch_init' in component.keys():
                    del component['scratch_init']
                if 'name' in component.keys():
                    del component['name']
                if 'id' in component.keys():
                    del component['id']
                if 'in' in component.keys():
                    del component['in']
                if 'in_y' in component.keys():
                    del component['in_y']
                if 'out' in component.keys():
                    del component['out']
                if 'main' in component.keys():
                    del component['main']
                if 'out' in component.keys():
                    del component['out']
                if 'fit_on' in component.keys():
                    del component['fit_on']

                comp_data['conf'] = component
                pipe['components'].append(comp_data)

            for name, val_ in conf['results'].items():
                pipe['res'][name] = val_
            pipelines.append(pipe)

    # Sorting pipelines
    pipelines = sort_old_pipes(pipelines, target_metric)

    # Create a format to use in the merged range.
    cell_format = workbook.add_format({'bold': 1,
                                       'border': 1,
                                       'align': 'center',
                                       'valign': 'vcenter'})

    # Start from the first cell. Rows and columns are zero indexed.
    row = 0
    col = 0
    # Write pipelines table
    for pipe in pipelines:
        h = write_old_pipe(worksheet, pipe, row, col, cell_format, max_com)
        row += h + 1

    # Write graphs (bars)
    workbook = plot_bar(workbook, pipelines, metrics)
    workbook.close()

    return None


# ________________________________________________Generate new table___________________________________________________
def get_data(log):
    dataset_names = {}
    max_com = 0

    for dataset_name, batch_val in log['experiments'].items():
        dataset_names[dataset_name] = []
        batch_sizes = {}
        for batch_size, models_val in batch_val.items():
            pipelines = []
            for model_name, val in models_val.items():
                for num, conf in val.items():
                    pipe = dict(index=int(num), components=[], res={})
                    # max amount of components
                    if max_com < len(conf['config']):
                        max_com = len(conf['config'])

                    for component in conf['config']:
                        comp_data = {}
                        comp_data['name'] = component.pop('component_name')

                        if 'save_path' in component.keys():
                            del component['save_path']
                        if 'load_path' in component.keys():
                            del component['load_path']
                        if 'scratch_init' in component.keys():
                            del component['scratch_init']
                        if 'name' in component.keys():
                            del component['name']
                        if 'id' in component.keys():
                            del component['id']
                        if 'in' in component.keys():
                            del component['in']
                        if 'in_y' in component.keys():
                            del component['in_y']
                        if 'out' in component.keys():
                            del component['out']
                        if 'main' in component.keys():
                            del component['main']
                        if 'out' in component.keys():
                            del component['out']
                        if 'fit_on' in component.keys():
                            del component['fit_on']

                        comp_data['conf'] = component
                        pipe['components'].append(comp_data)

                    for name, val_ in conf['results'].items():
                        pipe['res'][name] = val_
                    pipelines.append(pipe)
            batch_sizes[batch_size] = pipelines
        dataset_names[dataset_name].append(batch_sizes)

    return max_com, dataset_names


def write_info(sheet, num, target_metric, cell_format):
    # Start from the first cell. Rows and columns are zero indexed.
    # write info
    sheet.write(0, 0, "Number of pipelines:", cell_format)
    sheet.write(0, 1, num, cell_format)
    sheet.write(0, 2, "Target metric:", cell_format)
    sheet.write(0, 3, target_metric, cell_format)
    return 2, 0


def write_legend(sheet, row, col, metric_names, max_com, cell_format):
    # write legend
    sheet.write(row, col, "Pipeline", cell_format)
    sheet.merge_range(row, col + 1, row, max_com - 1, "Preprocessing", cell_format)
    sheet.write(row, max_com, "Model", cell_format)
    for k, met in enumerate(metric_names):
        sheet.write(row, max_com + 1 + k, met, cell_format)

    return row + 1, col


def write_dataset_name(sheet, sheet_2, row_1, row_2, col, name, dataset_list, format, max_l, target_metric,
                       metric_names):
    # write dataset name
    sheet.write(row_1, col, "Dataset name", format)
    sheet.write(row_1, col + 1, name, format)
    row_1 += 2

    # write dataset name
    sheet_2.write(row_2, col, "Dataset name", format)
    sheet_2.write(row_2, col + 1, name, format)
    row_2 += 2

    for bs_dict in dataset_list:
        row_1, row_2 = write_batch_size(row_1, row_2, col, bs_dict, sheet, sheet_2, format, max_l, target_metric,
                                        metric_names)

    return row_1 + 2, row_2 + 2


def write_batch_size(row1, row2, col, bs_dict, sheet, sheet_2, format, max_l, target_metric, metric_names):
    row_1 = row1
    row_2 = row2

    for bs, val_ in bs_dict.items():
        # write batch size
        sheet.write(row_1, col, "Batch size", format)
        sheet.write(row_1, col + 1, bs, format)
        row_1 += 1

        row_1, col = write_legend(sheet, row_1, col, metric_names, max_l, format)

        sheet_2.write(row_2, col, "Batch size", format)
        sheet_2.write(row_2, col + 1, bs, format)
        row_2 += 1

        row_2, col = write_legend(sheet_2, row_2, col, metric_names, max_l, format)

        # Write pipelines table
        row_1 = write_table(sheet, val_, row_1, col, format, max_l)
        # Get the best pipelines
        best_pipelines = get_best(val_, target_metric)
        # Sorting pipelines
        best_pipelines = sort_pipes(best_pipelines, target_metric)
        # Write sort pipelines table
        row_2 = write_table(sheet_2, best_pipelines, row_2, col, format, max_l, write_conf=False)

        row_1 += 2
        row_2 += 2

    row_1 += 2
    row_2 += 2

    return row_1, row_2


def write_metrics(sheet, comp_dict, start_x, start_y, cell_format):
    data_names = list(comp_dict['res'].keys())
    metric_names = list(comp_dict['res'][data_names[-1]].keys())

    for j, tp in enumerate(data_names):
        for k, met in enumerate(metric_names):
            sheet.write(start_x, start_y + k + 1, comp_dict['res'][tp][met], cell_format)
    return None


def write_config(sheet, comp_dict, x, y, cell_format):
    z = {}
    for i, comp in enumerate(comp_dict['components']):
        z[str(i)] = comp['conf']
    s = json.dumps(z)
    sheet.write(x, y, s, cell_format)
    return None


def write_pipe(sheet, pipe_dict, start_x, start_y, cell_format, max_, write_conf):
    """ Add pipeline to the table """
    data_names = list(pipe_dict['res'].keys())
    metric_names = list(pipe_dict['res'][data_names[-1]].keys())

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
    if write_conf:
        write_config(sheet, pipe_dict, x, max_ + len(metric_names) + 1, cell_format)
    return None


def write_table(worksheet, pipelines, row, col, cell_format, max_l, write_conf=True):
    # Write pipelines table
    for pipe in pipelines:
        write_pipe(worksheet, pipe, row, col, cell_format, max_l, write_conf)
        row += 1
    return row


def get_best(data, target):
    def get_name(pipeline):
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
            tp = list(pipe['res'].keys())[-1]
            buf[pipe_name] = {'ind': pipe['index'], 'target': pipe['res'][tp][target]}
        else:
            tp = list(pipe['res'].keys())[-1]
            if buf[pipe_name]['target'] <= pipe['res'][tp][target]:
                buf[pipe_name]['target'] = pipe['res'][tp][target]
                buf[pipe_name]['ind'] = pipe['index']

    for key, val in buf.items():
        inds.append(val['ind'])

    del buf

    for pipe in data:
        if pipe['index'] in inds:
            best_pipes.append(pipe)

    return best_pipes


def sort_pipes(pipes, target_metric):
    ind_val = []
    sort_pipes_ = []
    dtype = [('value', 'float'), ('index', 'int')]
    for pipe in pipes:
        if 'test' not in pipe['res'].keys():
            name = list(pipe['res'].keys())[0]
        else:
            name = 'test'
        rm = pipe['res'][name]
        ind_val.append((rm[target_metric], pipe['index']))

    ind_val = np.sort(np.array(ind_val, dtype=dtype), order='value')
    for com in ind_val:
        ind = com[1]
        for pipe in pipes:
            if pipe['index'] == ind:
                sort_pipes_.append(pipe)

    del pipes, ind_val

    sort_pipes_.reverse()

    return sort_pipes_


def build_pipeline_table(log_data, target_metric=None, save_path='./'):
    exp_name = log_data['experiment_info']['exp_name']
    date = log_data['experiment_info']['date']
    metrics = log_data['experiment_info']['metrics']
    num_p = log_data['experiment_info']['number_of_pipes']
    if target_metric is None:
        target_metric = metrics[0]

    # read data from log
    max_l, pipe_data = get_data(log_data)
    # create xlsx table form
    workbook = xlsxwriter.Workbook(join(save_path, 'Report_{0}_{1}.xlsx'.format(exp_name, date)))
    worksheet_1 = workbook.add_worksheet("Pipelines_sort")
    worksheet_2 = workbook.add_worksheet("Pipelines_table")
    # Create a cell format
    cell_format = workbook.add_format({'bold': 1,
                                       'border': 1,
                                       'align': 'center',
                                       'valign': 'vcenter'})
    # write legend to tables
    row, col = write_info(worksheet_1, num_p, target_metric, cell_format)
    row, col = write_info(worksheet_2, num_p, target_metric, cell_format)

    row1 = row
    row2 = row

    for dataset_name, dataset_dict in pipe_data.items():
        row1, row2 = write_dataset_name(worksheet_2, worksheet_1, row1, row2, col, dataset_name, dataset_dict,
                                        cell_format, max_l, target_metric, metrics)

    workbook.close()
    return None


# ___________________________________________________Generate plots___________________________________________________


def results_analizator(log, target_metric='f1_weighted', num_best=5):
    models_names = list(log['experiments'].keys())
    metrics = log['experiment_info']['metrics']
    if target_metric not in metrics:
        print("Warning: Target metric '{}' not in log. The fisrt metric in log will be use as target metric.".format(
            target_metric))
        target_metric = metrics[0]

    main = {'models': {}, 'best_model': {}}
    for name in models_names:
        main['models'][name] = {}
        for met in metrics:
            main['models'][name][met] = []
        main['models'][name]['pipe_conf'] = []

    for name in models_names:
        for key, val in log['experiments'][name].items():
            for met in metrics:
                if val['results'].get('test') is not None:
                    main['models'][name][met].append(val['results']['test']['metrics'][met])
                else:
                    main['models'][name][met].append(val['results']['valid']['metrics'][met])
            main['models'][name]['pipe_conf'].append(val['config'])

    m = 0
    mxname = ''
    best_pipeline = None
    sort_met = {}

    for name in models_names:
        sort_met[name] = {}
        for met in metrics:
            tmp = np.sort(main['models'][name][met])[-num_best:]
            sort_met[name][met] = tmp[::-1]
            if tmp[0] > m:
                m = tmp[0]
                mxname = name
                best_pipeline = main['models'][name]['pipe_conf'][main['models'][name][met].index(m)]

    main['sorted'] = sort_met

    main['best_model']['name'] = mxname
    main['best_model']['score'] = m
    main['best_model']['target_metric'] = target_metric
    main['best_model']['best_pipeline'] = best_pipeline

    return main


def plot_bar(book, data, metrics_):
    graph_worksheet = book.add_worksheet('Plots')
    chart = book.add_chart({'type': 'bar'})

    # TODO fix this constrains on numbers of colors
    colors = ['#FF9900', '#e6ff00', '#ff1a00', '#00ff99', '#9900ff', '#0066ff']
    pipe_ind = []
    metric = {}
    x = 0
    y = 0
    for met in metrics_:
        metric[met] = []
        for pipe in data:
            ind = int(pipe['index'])
            if ind not in pipe_ind:
                pipe_ind.append(ind)

            if 'test' not in pipe['res']:
                name_ = list(pipe['res'].keys())[0]
                metric[met].append(pipe['res'][name_][met])
            else:
                metric[met].append(pipe['res']['test'][met])

        # write in book
        graph_worksheet.merge_range(x, y, x, y+1, met)
        graph_worksheet.write_column(x+1, y, pipe_ind)
        graph_worksheet.write_column(x+1, y+1, metric[met])

        # Add a series to the chart.
        chart.add_series({'name': ['Plots', x, y, x, y+1],
                          'categories': ['Plots', x+1, y, x+len(pipe_ind), y],
                          'values': ['Plots', x+1, y+1, x+len(pipe_ind), y+1],
                          'data_labels': {'value': True, 'legend_key': True, 'position': 'center', 'leader_lines': True,
                                          'num_format': '#,##0.00', 'font': {'name': 'Consolas'}},
                          'border': {'color': 'black'},
                          'fill': {'colors': colors}})

        y += 2

    # Add a chart title and some axis labels.
    chart.set_title({'name': 'Results bar'})
    chart.set_x_axis({'name': 'Scores'})
    chart.set_y_axis({'name': 'Pipelines'})

    # Set an Excel chart style.
    chart.set_style(11)

    # Insert the chart into the worksheet.
    graph_worksheet.insert_chart('H1', chart)  # , {'x_offset': 25, 'y_offset': 10}
    return book


def plot_res(info, save=True, savepath='./', width=0.2, fheight=8, fwidth=12, ext='png'):
    # prepeare data
    info = info['sorted']

    bar_list = []
    models = list(info.keys())
    metrics = list(info[models[0]].keys())
    n = len(metrics)

    # print(models)

    for met in metrics:
        tmp = []
        for model in models:
            tmp.append(info[model][met][0])
        bar_list.append(tmp)

    x = np.arange(len(models))

    # ploting
    fig, ax = plt.subplots()
    fig.set_figheight(fheight)
    fig.set_figwidth(fwidth)

    colors = plt.cm.Paired(np.linspace(0, 0.5, len(bar_list)))
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Scores').set_fontsize(20)
    ax.set_title('Scores by metric').set_fontsize(20)

    bars = []
    for i, y in enumerate(bar_list):
        if i == 0:
            bars.append(ax.bar(x, y, width, color=colors[i]))
        else:
            bars.append(ax.bar(x + i*width, y, width, color=colors[i]))

    # plot x sticks and labels
    ax.set_xticks(x - width / 2 + n * width / 2)
    ax.set_xticklabels(tuple(models), fontsize=15)

    yticks = ax.get_yticks()
    ax.set_yticklabels(['{0:.2}'.format(float(y)) for y in yticks], fontsize=15)

    ax.grid(True, linestyle='--', color='b', alpha=0.1)

    # plot legend
    # ax.legend(tuple([bar[0] for bar in bars]), tuple(metrics), loc='upper left', bbox_to_anchor=(1, 1))
    ax.legend(tuple([bar[0] for bar in bars]), tuple(metrics))

    # auto lables
    def autolabel(columns):
        for rects in columns:
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '{0:.2}'.format(float(height)),
                        ha='center', va='bottom', fontsize=12)

    autolabel(bars)
    plt.ylim(0, 1.1)

    # show the picture
    if not save:
        plt.show()
    else:
        if not isdir(savepath):
            mkdir(savepath)
        adr = join(savepath, '{0}.{1}'.format('main_hist', ext))
        fig.savefig(adr, dpi=100)
        plt.close(fig)

    return None


# _________________________________________________Built report_______________________________________________________


def results_visualization(root, savepath, plot, target_metric=None):
    with open(join(root, root.split('/')[-1] + '.json'), 'r') as log_file:
        log = json.load(log_file)
        log_file.close()

    # create the xlsx file with results of experiments
    build_pipeline_table(log, target_metric=target_metric, save_path=root)
    # build_report(log, target_metric=target_metric, save_path=root)
    if plot:
        # scrub data from log for image creating
        info = results_analizator(log, target_metric=target_metric)
        # plot histogram
        plot_res(info, savepath=savepath)

    return None
