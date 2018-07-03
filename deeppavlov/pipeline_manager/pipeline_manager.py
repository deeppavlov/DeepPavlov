from time import time
from datetime import datetime
from os.path import join

from deeppavlov.pipeline_manager.pipegen import PipeGen
from deeppavlov.pipeline_manager.train import train_evaluate_model_from_dict

# from .logger import Logger
# from .watcher import Watcher
# from .utils import results_visualization


def normal_time(z):
    if z > 1:
        h = z/3600
        m = z % 3600/60
        s = z % 3600 % 60
        t = '%i:%i:%i' % (h, m, s)
    else:
        t = '{0:.2}'.format(z)
    return t


class PipelineManager:
    def __init__(self, config_path, exp_name, mode='train',
                 info=None,
                 root='./experiments/',
                 hyper_search='grid',
                 sample_num=10,
                 add_watcher=False,
                 target_metric=None):

        self.config_path = config_path
        self.exp_name = exp_name
        self.mode = mode
        self.info = info
        self.hyper_search = hyper_search
        self.sample_num = sample_num
        self.date = datetime.now()
        self.add_watcher = add_watcher
        self.target_metric = target_metric
        self.pipeline_generator = None

        self.root = root
        self.save_path = join(self.root, '{}-{}-{}'.format(self.date.year, self.date.month, self.date.day),
                              self.exp_name, 'checkpoints')

        # self.logger = Logger(exp_name, root, self.info, self.date)
        # self.start_exp = time()
        # self.logger.log['experiment_info']['metrics'] = self.metrics
        # self.logger.log['experiment_info']['target_metric'] = self.target_metric

    def run(self):
        # best_models = {}
        # create the pipeline generator
        self.pipeline_generator = PipeGen(self.config_path, n=self.sample_num, stype=self.hyper_search)

        # Start generating pipelines configs
        print('[ Experiment start - {0} pipes, will be run]'.format(self.pipeline_generator.len))
        exp_start_time = time()
        for i, pipe in enumerate(self.pipeline_generator()):
            # print progress
            if i != 0:
                itime = normal_time(((time() - exp_start_time) / i) * (self.pipeline_generator.len - i))
                print('\n')
                print('[ Progress: pipe {0}/{1}; Time left: {2}; ]'.format(i+1, self.pipeline_generator.len, itime))

            # self.logger.pipe_ind = i
            # pipe_start = time()
            # # add watcher if need
            # if self.add_watcher:
            #     watcher = Watcher(join(self.root, '{0}-{1}-{2}'.format(self.date.year, self.date.month, self.date.day),
            #                            self.exp_name), self.seed)

            if self.mode == 'train':
                train_evaluate_model_from_dict(pipe, to_train=True, to_validate=True)
            elif self.mode == 'evaluate':
                train_evaluate_model_from_dict(pipe, to_train=False, to_validate=False)

            # save best models
            # self.logger.pipe_time = normal_time(time() - pipe_start)
            # self.logger.pipe_res = dataset_i['results']
            # self.logger.get_pipe_log()

        # save log
        # self.logger.log['experiment_info']['full_time'] = normal_time(time() - self.start_exp)
        # self.logger.save()

        # visualization of results
        # path = join(self.root, '{0}-{1}-{2}'.format(self.date.year, self.date.month, self.date.day), self.exp_name)
        # results_visualization(path, join(path, 'results', 'images'), self.target_metric)

        return None
