import xlsxwriter
import json


path_to_log = '/home/mks/projects/DeepPavlov/experiments/2018-7-6/plot_test/plot_test.json'


def build_report(log):
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

    workbook = xlsxwriter.Workbook('Report_{0}_{1}.xlsx'.format(exp_name, date))
    worksheet = workbook.add_worksheet("Pipelines_table")
    #################################################################################################
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
                elif 'load_path' in component.keys():
                    del component['load_path']
                elif 'scratch_init' in component.keys():
                    del component['scratch_init']

                comp_data['conf'] = component
                pipe['components'].append(comp_data)

            for name, val_ in conf['results'].items():
                pipe['res'][name] = val_['metrics']
            pipelines.append(pipe)
    #################################################################################################

    #################################################################################################
    def get_pipe_sq(pipe_dict):
        names = pipe_dict['res'].keys()
        m = 0
        for comp in pipe_dict['components']:
            if m < len(comp['conf']):
                m = len(comp['conf'])

        height = 1 + m
        width = 1 + 2 * len(pipe_dict['components']) + len(pipe_dict['res']) * len(pipe_dict['res'][names[0]])
        return height, width

    def write_component(sheet, comp_dict, start_x, start_y, mx_height, cell_format):
        delta = mx_height - len(comp_data['conf'])
        sheet.merge_range(start_x, start_y, start_x, start_y + 1, comp_dict['name'], cell_format=cell_format)

        x = start_x + 1
        for par, value in comp_data['conf'].items():
            sheet.write(x, start_y, par, cell_format=cell_format)
            sheet.write(x, start_y+1, value, cell_format=cell_format)
            x += 1

        for i in range(delta):
            sheet.write(x, start_y, '-', cell_format=cell_format)
            sheet.write(x, start_y + 1, '-', cell_format=cell_format)
            x += 1

        return None
    def write_metrics(sheet, comp_dict, start_x, start_y, mx_height, cell_format):
        sheet.merge_range(start_x, start_y, start_x, start_y + len(comp_dict['res']), 'Metrics',
                          cell_format=cell_format)

        return None

    def write_pipe(sheet, pipe_dict, start_x, start_y, cell_format):
        height, width = get_pipe_sq(pipe_dict)
        height = height - 1

        sheet.write(start_x, start_y, 'index', cell_format=cell_format)
        sheet.merge_range(start_x + 1, start_y, start_x + height, start_y, pipe_dict['index'],
                          cell_format=cell_format)

        for conf in pipe_dict['components']:




        return None

    #################################################################################################

    # Create a format to use in the merged range.
    cell_format = workbook.add_format({'bold': 1,
                                       'border': 1,
                                       'align': 'center',
                                       'valign': 'vcenter'})

    # Some data we want to write to the worksheet.
    expenses = (
        ['Rent', 1000],
        ['Gas', 100],
        ['Food', 300],
        ['Gym', 50],
    )

    # Start from the first cell. Rows and columns are zero indexed.
    row = 0
    col = 0

    # Iterate over the data and write it out row by row.
    for item, cost in expenses:
        worksheet.write(row, col, item)
        worksheet.write(row, col + 1, cost)
        row += 1

    # Write a total using a formula.
    worksheet.write(row, 0, 'Total')
    worksheet.write(row, 1, '=SUM(B1:B4)')

    worksheet.merge_range()

    workbook.close()

    return None
