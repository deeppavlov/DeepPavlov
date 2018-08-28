import json
import os

from collections import OrderedDict
from os.path import join, isdir, isfile
from deeppavlov.pipeline_manager.utils import normal_time


class Logger(object):
    """
    The class implements data collection on how the experiment is going. Pipeline configuration information,
    pipeline results, and time information is collected.
    """
    def __init__(self, name, root, info, date, plot):
        """
        Init log, and creates folders for logs, report and checkpoints.

        Args:
            name: str; name of the experiments.
            root: str; path to root folder.
            info: dict; ome additional information that you want to add to the log, the content of the dictionary
             does not affect the algorithm
            date: str; date of the experiment.
        """
        self.exp_name = name
        self.exp_inf = info
        self.root = root
        self.date = date

        # tmp parameters
        self.pipe_ind = 0
        self.pipe_conf = None
        self.model = None
        self.pipe_res = None
        self.pipe_time = None

        # build folder dependencies
        self.log_path = join(self.root, date, self.exp_name)
        self.log_file = join(self.log_path, self.exp_name + '.json')

        if not isdir(self.log_path):
            os.makedirs(self.log_path)
        if plot:
            if not isdir(join(self.log_path, 'images')):
                os.makedirs(join(self.log_path, 'images'))

        self.log = OrderedDict(experiment_info=OrderedDict(date=date,
                                                           exp_name=self.exp_name,
                                                           root=self.root,
                                                           info=self.exp_inf,
                                                           number_of_pipes=None),
                               dataset={},
                               experiments=OrderedDict())

    def tmp_reset(self):
        # tmp parameters
        self.pipe_ind = 0
        self.pipe_conf = None
        self.model = None
        self.pipe_res = None
        self.pipe_time = None

    def save(self):
        """save log in file"""
        if not isfile(self.log_file):
            with open(self.log_file, 'w') as log_file:
                json.dump(self.log, log_file)
                log_file.close()
        else:
            with open(self.log_file, 'r') as old_file:
                old_log = json.load(old_file)
                old_file.close()

            self.log = self.merge_logs(old_log, self.log)
            with open(self.log_file, 'w') as log_file:
                json.dump(self.log, log_file)
                log_file.close()

    @staticmethod
    def merge_logs(old_log, new_log):
        """ Combines two logs into one """
        new_models_names = list(new_log['experiments'].keys())

        for name in new_models_names:
            if name not in old_log['experiments'].keys():
                old_log['experiments'][name] = new_log['experiments'][name]
            else:
                old_npipe = len(old_log['experiments'][name])  # - 1
                k = 0
                for nkey, nval in new_log['experiments'][name].items():
                    match = False
                    for okey, oval in old_log['experiments'][name].items():
                        if nval['config'] == oval['config']:
                            old_log['experiments'][name][okey] = new_log['experiments'][name][nkey]
                            match = True
                        else:
                            pass

                    if not match:
                        k += 1
                        old_log['experiments'][name][str(old_npipe+k)] = new_log['experiments'][name][nkey]

        # addition time
        t_old = old_log['experiment_info']['full_time'].split(':')
        t_new = new_log['experiment_info']['full_time'].split(':')
        sec = int(t_old[2]) + int(t_new[2]) + (int(t_old[1]) + int(t_new[1]))*60 + (int(t_old[0]) + int(t_new[0]))*3600

        old_log['experiment_info']['full_time'] = normal_time(sec)

        return old_log

    def get_pipe_log(self):
        """ Updates the log with information about the new pipeline """
        ops_times = {}

        if (self.model is None) and (self.pipe_conf is not None):
            for component in self.pipe_conf:
                if component.get('main') is True:
                    self.model = component['name']

        pipe_name = '-->'.join([x['name'] for x in self.pipe_conf])

        if self.model not in self.log['experiments'].keys():
            self.log['experiments'][self.model] = OrderedDict()
            self.log['experiments'][self.model][self.pipe_ind] = {'config': self.pipe_conf,
                                                                  'light_config': pipe_name,
                                                                  'time': self.pipe_time,
                                                                  'ops_time': ops_times,
                                                                  'results': self.pipe_res}
        else:
            self.log['experiments'][self.model][self.pipe_ind] = {'config': self.pipe_conf,
                                                                  'light_config': pipe_name,
                                                                  'time': self.pipe_time,
                                                                  'ops_time': ops_times,
                                                                  'results': self.pipe_res}

        self.tmp_reset()
        return self
