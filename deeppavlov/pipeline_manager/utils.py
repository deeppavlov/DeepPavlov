import pandas as pd
import numpy as np
import json
import xlsxwriter
import matplotlib
import matplotlib.pyplot as plt  # .pyplot as plt

from os.path import join, isdir
from os import mkdir
matplotlib.use('agg')


def normal_time(z):
    if z > 1:
        h = z/3600
        m = z % 3600/60
        s = z % 3600 % 60
        t = '%i:%i:%i' % (h, m, s)
    else:
        t = '{0:.2}'.format(z)
    return t


# ------------------------------------------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------------------------------------------


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


def get_table(log, savepath, target_metric='f1_weighted', num_best=None):
    metrics = log['experiment_info']['metrics']
    if target_metric not in metrics:
        print("Warning: Target metric '{}' not in log. The fisrt metric in log will be use as target metric.".format(
            target_metric))
        target_metric = metrics[0]

    pd_main = {"Model": [], "Speller": [], "Tokenizer": [], "Lemmatizer": [], "Vectorizer": []}
    for met in metrics:
        pd_main[met] = []

    for name in log['experiments'].keys():
        for key, val in log['experiments'][name].items():
            pd_main['Model'].append(name)
            ops = val['light_config'].split('-->')

            for met in metrics:
                pd_main[met].append(val['results']['valid']['metrics'][met])

            spel = False
            tok = False
            lem = False
            vec = False

            for op in ops:
                op_name, op_type = op.split('_')
                if op_type == 'Speller':
                    pd_main["Speller"].append(op_name)
                    spel = True
                elif op_type == 'Tokenizer':
                    pd_main["Tokenizer"].append(op_name)
                    tok = True
                elif op_type == 'Lemmatizer':
                    pd_main["Lemmatizer"].append(op_name)
                    lem = True
                elif op_type == 'vectorizer':
                    pd_main["Vectorizer"].append(op_name)
                    vec = True
                else:
                    pass

            if not spel:
                pd_main["Speller"].append("None")
            if not tok:
                pd_main["Tokenizer"].append("None")
            if not lem:
                pd_main["Lemmatizer"].append("None")
            if not vec:
                pd_main["Vectorizer"].append("None")

    pdf = pd.DataFrame(pd_main)
    pdf = pdf.sort_values(target_metric, ascending=False)

    # get slice if need
    if num_best is not None:
        pdf = pdf[:num_best+1]

    #     pt = pd.pivot_table(pdf, index=["Speller", "Tokenizer", "Lemmatizer", "Vectorizer", "Model"])
    pt = pd.pivot_table(pdf,
                        index=["Model", "Speller", "Tokenizer", "Lemmatizer", "Vectorizer"])
    pt = pt.reindex(pt.sort_values(by=target_metric, ascending=False).index)

    # save it as excel
    writer = pd.ExcelWriter(join(savepath, 'report.xlsx'))
    pt.to_excel(writer, 'Sheet1')
    writer.save()
    return None


def ploting_hist(x, y, plot_name='Plot', color='y', width=0.35, plot_size=(10, 6), axes_names=['X', 'Y'],
                 x_lables=None, y_lables=None, xticks=True, legend=True, ext='png', savepath='./results/images/'):
    fig, ax = plt.subplots(figsize=plot_size)
    rects = ax.bar(x, y, width, color=color)

    # add some text for labels, title and axes ticks
    ax.set_xlabel(axes_names[0])
    ax.set_ylabel(axes_names[1])
    ax.set_title(plot_name)

    if xticks and x_lables is not None:
        ax.set_xticks(x)
        ax.set_xticklabels(x_lables)

    if legend and y_lables is not None:
        ax.legend((rects[0],), y_lables)

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.01 * height,
                    '{0:.3}'.format(float(height)),
                    ha='center', va='bottom')

    autolabel(rects)

    if not isdir(savepath):
        mkdir(savepath)
    adr = join(savepath, '{0}.{1}'.format(plot_name, ext))
    fig.savefig(adr, dpi=200)
    fig.close()
    return None


def plot_res_table(info, save=True, savepath='./', width=0.2, fheight=8, fwidth=12, ext='png'):
    # prepeare data
    info = info['sorted']
    bar_list = []
    models = list(info.keys())
    metrics = list(info[models[0]].keys())
    n = len(metrics)

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
            bars.append(ax.bar(x + i * width, y, width, color=colors[i]))

    yticks = ax.get_yticks()
    ax.set_yticklabels(['{0:.2}'.format(float(y)) for y in yticks], fontsize=15)

    ax.grid(True, linestyle='--', color='b', alpha=0.1)

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n):
        cell_text.append(['test' for x in models])

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=metrics,
                          rowColours=colors,
                          colLabels=models,
                          loc='bottom')

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    # plot x sticks and labels
    plt.xticks([])

    # add some text for labels, title and axes ticks
    ax.set_ylabel('Scores')
    ax.set_title('Scores by metric')

    # plot legend
    ax.legend(tuple([bar[0] for bar in bars]), tuple(metrics))

    # auto lables
    def autolabel(columns):
        """
        Attach a text label above each bar displaying its height
        """
        for rects in columns:
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '{0:.2}'.format(float(height)),
                        ha='center', va='bottom', fontsize=12)

    autolabel(bars)
    plt.ylim(0, 1.1)

    # show the picture
    if save:
        if not isdir(savepath):
            mkdir(savepath)
        adr = join(savepath, '{0}.{1}'.format('main_hist_tab', ext))
        fig.savefig(adr, dpi=100)
        plt.close(fig)
    else:
        plt.show()
    return None


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
        """
        Attach a text label above each bar displaying its height
        """
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


def results_visualization(root, savepath, target_metric=None):
    with open(join(root, root.split('/')[-1] + '.json'), 'r') as log_file:
        log = json.load(log_file)
        log_file.close()

    # reading and scrabbing data
    info = results_analizator(log, target_metric=target_metric)
    plot_res(info, savepath=savepath)
    # plot_res_table(info, savepath=savepath)
    # get_table(log, target_metric=target_metric, savepath=join(root, 'results'))

    return None
