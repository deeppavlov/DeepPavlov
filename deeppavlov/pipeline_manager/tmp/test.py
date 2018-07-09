import xlsxwriter
import json


path_to_log = '/home/mks/projects/DeepPavlov/experiments/2018-7-6/plot_test/plot_test.json'


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
    sheet.merge_range(start_x, start_y, start_x, start_y + 1, comp_dict['name'], cell_format=cell_format)

    x = start_x + 1
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
    for j, tp in enumerate(comp_dict['res'].keys()):
        sheet.merge_range(start_x, start_y + j*len(comp_dict['res']), start_x,
                          start_y + j*len(comp_dict['res']) + len(comp_dict['res']), tp, cell_format=cell_format)
        for k, met in enumerate(comp_dict['res'][tp].keys()):
            sheet.merge_range(start_x + 1, start_y + j*len(comp_dict['res']), start_x + mx_height,
                              start_y + j*len(comp_dict['res']) + len(comp_dict['res']),
                              comp_dict['res'][tp][met], cell_format=cell_format)
    return None


def write_pipe(sheet, pipe_dict, start_x, start_y, cell_format):
    height, width = get_pipe_sq(pipe_dict)
    height = height - 1

    sheet.write(start_x, start_y, 'pipeline_index', cell_format)
    sheet.merge_range(start_x + 1, start_y, start_x + height, start_y, pipe_dict['index'],
                      cell_format=cell_format)

    x = start_x
    y = start_y + 1
    for conf in pipe_dict['components']:
        write_component(sheet, conf, x, y, height, cell_format)
        y += 2

    write_metrics(sheet, pipe_dict, x, y, height, cell_format)

    return None


# start
with open(path_to_log, 'r') as lgd:
    log_data = json.load(lgd)

exp_name = log_data['experiment_info']['exp_name']
date = log_data['experiment_info']['date']

workbook = xlsxwriter.Workbook('Report_{0}_{1}.xlsx'.format(exp_name, date))
bold = workbook.add_format({'bold': True})
worksheet = workbook.add_worksheet("Pipelines_table")
cell_format = workbook.add_format({'bold': 1, 'border': 1, 'align': 'center', 'valign': 'vcenter'})

row = 2
col = 2

pipelines = []
for model_name, val in log_data['experiments'].items():
    for num, conf in val.items():
        pipe = dict(index=int(num), components=[], res={})
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

# test
write_pipe(worksheet, pipelines[0], row, col, cell_format)
workbook.close()
