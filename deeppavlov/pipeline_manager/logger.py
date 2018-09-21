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
        self.batch_size = None
        self.dataset = None

        # build folder dependencies
        self.log_path = join(self.root, date, self.exp_name)
        self.log_file = join(self.log_path, self.exp_name + '.json')

        if not isdir(self.log_path):
            os.makedirs(self.log_path)
        if plot:
            if not isdir(join(self.log_path, 'images')):
                os.makedirs(join(self.log_path, 'images'))

        self.old_num_pipes = None
        self.old_full_time = None

        if isfile(self.log_file):
            with open(self.log_file, 'r') as log_file:
                self.log = json.load(log_file)
                log_file.close()

            if "number_of_pipes" in self.log['experiment_info'].keys() and \
                    (self.log['experiment_info']['number_of_pipes'] is not None):
                self.old_num_pipes = self.log['experiment_info']['number_of_pipes']
                self.log['experiment_info']['number_of_pipes'] = None
            if "full_time" in self.log['experiment_info'].keys() and \
                    (self.log['experiment_info']['full_time'] is not None):
                self.old_full_time = self.log['experiment_info']['full_time']
                self.log['experiment_info']['full_time'] = None
        else:
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
        self.batch_size = None
        self.dataset = None
        self.pipe_conf = None
        self.model = None
        self.pipe_res = None
        self.pipe_time = None

    def save(self):
        """save log in file"""
        with open(self.log_file, 'w') as log_file:
            json.dump(self.log, log_file)
            log_file.close()

    def update_times(self):
        # update time
        if self.old_full_time:
            t_old = self.old_full_time.split(':')
            t_new = self.log['experiment_info']['full_time'].split(':')
            sec = int(t_old[2]) + int(t_new[2]) + (int(t_old[1]) + int(t_new[1])) * 60 + (
                    int(t_old[0]) + int(t_new[0])) * 3600
            self.log['experiment_info']['full_time'] = normal_time(sec)

    def update_pipes(self):
        # update num of pipes
        if self.old_num_pipes:
            n_old = int(self.old_num_pipes)
            n_new = int(self.log['experiment_info']['number_of_pipes'])
            self.log['experiment_info']['number_of_pipes'] = n_old + n_new

    @staticmethod
    def merge_logs(old_log, new_log):
        """ Combines two logs into one """

        # update time
        t_old = old_log['experiment_info']['full_time'].split(':')
        t_new = new_log['experiment_info']['full_time'].split(':')
        sec = int(t_old[2]) + int(t_new[2]) + (int(t_old[1]) + int(t_new[1])) * 60 + (
                    int(t_old[0]) + int(t_new[0])) * 3600
        old_log['experiment_info']['full_time'] = normal_time(sec)
        # update num of pipes
        n_old = int(old_log['experiment_info']['number_of_pipes'])
        n_new = int(new_log['experiment_info']['number_of_pipes'])
        old_log['experiment_info']['number_of_pipes'] = n_old + n_new

        for dataset_name, dataset_val in new_log['experiments'].items():
            if dataset_name not in old_log['experiments'].keys():
                old_log['experiments'][dataset_name] = dataset_val
            else:
                for batch, batch_val in dataset_val.items():
                    if batch not in old_log['experiments'][dataset_name].keys():
                        old_log['experiments'][dataset_name][batch] = batch_val
                    else:
                        for name, val in batch_val.items():
                            if name not in old_log['experiments'][dataset_name][batch].keys():
                                old_log['experiments'][dataset_name][batch][name] = val
                            else:
                                for nkey, nval in new_log['experiments'][dataset_name][batch][name].items():
                                    match = False
                                    for okey, oval in old_log['experiments'][dataset_name][batch][name].items():
                                        if nval['config'] == oval['config']:
                                            match = True
                                    if not match:
                                        n_old += 1
                                        old_log['experiments'][dataset_name][batch][name][str(n_old)] = \
                                            new_log['experiments'][dataset_name][batch][name][nkey]

        return old_log

    def get_pipe_log(self):
        """ Updates the log with information about the new pipeline """
        ops_times = {}

        if (self.model is None) and (self.pipe_conf is not None):
            for component in self.pipe_conf:
                if component.get('main') is True:
                    self.model = component['component_name']

        pipe_name = '-->'.join([x['component_name'] for x in self.pipe_conf])

        if self.dataset not in self.log['experiments'].keys():
            self.log['experiments'][self.dataset] = OrderedDict()

        if self.model not in self.log['experiments'][self.dataset].keys():
            self.log['experiments'][self.dataset][self.model] = OrderedDict()
            self.log['experiments'][self.dataset][self.model][self.pipe_ind] =\
                {'config': self.pipe_conf,
                 'light_config': pipe_name,
                 'time': self.pipe_time,
                 'batch_size': self.batch_size,
                 'ops_time': ops_times,
                 'results': self.pipe_res}
        else:
            self.log['experiments'][self.dataset][self.model][self.pipe_ind] =\
                {'config': self.pipe_conf,
                 'light_config': pipe_name,
                 'time': self.pipe_time,
                 'batch_size': self.batch_size,
                 'ops_time': ops_times,
                 'results': self.pipe_res}

        self.tmp_reset()
        return self
