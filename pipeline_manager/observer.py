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
from pipeline_manager.utils import merge_logs


class Observer(object):
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

        self.old_log = None
        if isfile(self.log_file):
            with open(self.log_file, 'r') as log_file:
                self.old_log = json.load(log_file)
                log_file.close()

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

    def write(self):
        """save log in file"""
        with open(self.log_file, 'w') as log_file:
            json.dump(self.log, log_file)
            log_file.close()

    def save(self):
        if self.old_log is not None:
            self.log = merge_logs(self.old_log, self.log)
        with open(self.log_file, 'w') as log_file:
            json.dump(self.log, log_file)
            log_file.close()

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
