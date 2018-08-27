from time import time
from datetime import datetime
from os.path import join
from copy import copy
from shutil import rmtree

from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.pipeline_manager.pipegen import PipeGen
from deeppavlov.pipeline_manager.utils import normal_time
from deeppavlov.pipeline_manager.logger import Logger
from deeppavlov.pipeline_manager.utils import results_visualization


class PipelineManager:
    """
    The class implements functions for automatic iteration of pipelines and hyperparameters search.
    """
    def __init__(self, config_path, exp_name, date=None, mode='train', info=None, root='./experiments/',
                 hyper_search='grid', sample_num=10, target_metric=None):
        """
        Initialize logger, builds a directory tree, initialize date.

        Args:
            config_path: str; path to config file.
            exp_name: str; name of the experiment.
            date: str; date of the experiment.
            mode: str; train or evaluate - the trigger that determines the operation of the algorithm
            info: dict; some additional information that you want to add to the log, the content of the dictionary
             does not affect the algorithm
            root: str; root path, the root path where the report will be generated and saved checkpoints
            hyper_search: str; grid or random - the trigger that determines type of hypersearch
            sample_num: int; determines the number of generated pipelines, if hyper_search == random.
            target_metric: str; The metric name on the basis of which the results will be sorted when the report
             is generated. The default value is None, in this case the target metric is taken  the first name from
             those names that are specified in the config file. If the specified metric is not contained in DeepPavlov
             will be called error.
        """

        self.config_path = config_path
        self.exp_name = exp_name
        self.mode = mode
        self.info = info
        self.hyper_search = hyper_search
        self.sample_num = sample_num
        self.target_metric = target_metric
        self.pipeline_generator = None
        if date is not None:
            self.date = date
        else:
            date_ = datetime.now()
            self.date = '{}-{}-{}'.format(date_.year, date_.month, date_.day)

        self.root = root
        self.save_path = join(self.root, self.date, self.exp_name, 'checkpoints')

        self.logger = Logger(exp_name, root, self.info, self.date)
        self.start_exp = time()
        # start test
        self.test()

    def run(self):
        """
        Initializes the pipeline generator and runs the experiment. Creates a report after the experiments

        Returns:
            None
        """
        # create the pipeline generator
        self.pipeline_generator = PipeGen(self.config_path, self.save_path, n=self.sample_num, stype=self.hyper_search)

        # Start generating pipelines configs
        print('[ Experiment start - {0} pipes, will be run]'.format(self.pipeline_generator.len))
        exp_start_time = time()
        for i, pipe in enumerate(self.pipeline_generator()):
            # print progress
            if i != 0:
                itime = normal_time(((time() - exp_start_time) / i) * (self.pipeline_generator.len - i))
                print('\n')
                print('[ Progress: pipe {0}/{1}; Time left: {2}; ]'.format(i+1, self.pipeline_generator.len, itime))

            self.logger.log['experiment_info']['metrics'] = copy(pipe['train']['metrics'])
            self.logger.log['experiment_info']['target_metric'] = self.target_metric

            self.logger.pipe_ind = i + 1
            self.logger.pipe_conf = copy(pipe['chainer']['pipe'])

            # start pipeline time
            pipe_start = time()

            if self.mode == 'train':
                results = train_evaluate_model_from_config(pipe, to_train=True, to_validate=True)
            elif self.mode == 'evaluate':
                results = train_evaluate_model_from_config(pipe, to_train=False, to_validate=False)
            else:
                raise ValueError("Only 'train' and 'evaluate' mode are available, but {0} was found.".format(self.mode))

            # save best models
            self.logger.pipe_time = normal_time(time() - pipe_start)
            self.logger.pipe_res = results
            self.logger.get_pipe_log()

        # save log
        self.logger.log['experiment_info']['full_time'] = normal_time(time() - self.start_exp)
        self.logger.save()

        # visualization of results
        path = join(self.root, self.date, self.exp_name)
        results_visualization(path, join(path, 'results', 'images'), self.target_metric)
        return None

    def test(self):
        """
        Initializes the pipeline generator with tiny data and runs the test of experiment.

        Returns:
            None
        """
        # create tmp folder in self.save_path

        # create the pipeline generator
        pipeline_generator = PipeGen(self.config_path, self.save_path, n=self.sample_num, stype=self.hyper_search,
                                     test_mode=True)

        # Start generating pipelines configs
        print('[ Test start - {0} pipes, will be run]'.format(pipeline_generator.len))
        exp_start_time = time()
        for i, pipe in enumerate(pipeline_generator()):
            # print progress
            if i != 0:
                itime = normal_time(((time() - exp_start_time) / i) * (pipeline_generator.len - i))
                print('\n')
                print('[ Test progress: pipe {0}/{1}; Time left: {2}; ]'.format(i + 1, pipeline_generator.len, itime))

            if pipe['dataset_reader']['name'] == 'basic_classification_reader':
                pipe['dataset_reader'] = {
                    "name": "basic_classification_reader",
                    "x": "text",
                    "y": "target",
                    "data_path": '../tests/test_data/classification_data/'
                }
            else:
                raise ConfigError("Dataset reader is not intended for classification task."
                                  "Name of dataset_reader must be 'basic_classification_reader',"
                                  "but {} was found in config.".format(pipe['dataset_reader']['name']))

            if self.mode == 'train':
                results = train_evaluate_model_from_config(pipe, to_train=True, to_validate=False)
            elif self.mode == 'evaluate':
                results = train_evaluate_model_from_config(pipe, to_train=False, to_validate=False)
            else:
                raise ValueError("Only 'train' and 'evaluate' mode are available, but {0} was found.".format(self.mode))

        # del all tmp files in save path
        rmtree(join(self.save_path, "tmp"))

        print('[ The test was successful ]')
        return None
