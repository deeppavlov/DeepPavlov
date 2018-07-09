import xlsxwriter
import json
import numpy as np


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


def write_metrics(sheet, comp_dict, start_x, start_y, mx_height, cell_format):
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


def write_pipe(sheet, pipe_dict, start_x, start_y, cell_format, max_):
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
    write_metrics(sheet, pipe_dict, x, y + dy, height, cell_format)

    return height


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


def build_report(log, target_metric=None):
    if isinstance(log, str):
        with open(log, 'r') as lgd:
            log_data = json.load(lgd)
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

    workbook = xlsxwriter.Workbook('Report_{0}_{1}.xlsx'.format(exp_name, date))
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
                if component.get('id') is not None:
                    comp_data['name'] = component.pop('id')
                else:
                    comp_data['name'] = component.pop('name')

                if 'save_path' in component.keys():
                    del component['save_path']
                if 'load_path' in component.keys():
                    del component['load_path']
                if 'scratch_init' in component.keys():
                    del component['scratch_init']

                comp_data['conf'] = component
                pipe['components'].append(comp_data)

            for name, val_ in conf['results'].items():
                pipe['res'][name] = val_['metrics']
            pipelines.append(pipe)

    # Sorting pipelines
    pipelines = sort_pipes(pipelines, target_metric)

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
        h = write_pipe(worksheet, pipe, row, col, cell_format, max_com)
        row += h + 1

    # Write graphs (bars)
    workbook = plot_bar(workbook, pipelines, metrics)
    workbook.close()

    return None


path_to_log = '/home/mks/projects/DeepPavlov/experiments/2018-7-6/plot_test/plot_test.json'
build_report(path_to_log, target_metric='classification_f1')
