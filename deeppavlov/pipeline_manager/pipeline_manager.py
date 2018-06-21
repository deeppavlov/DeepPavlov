import os
from time import time
from datetime import datetime
from os.path import join, isdir
from copy import deepcopy
from sklearn.externals import joblib


class PipelineManager(object):
    def __init__(self, config_path, exp_name,
                 root='',
                 seed=42,
                 hyper_search='grid',
                 sample_num=10,
                 metrics=None,
                 target_metric=None):

        self.config_path = config_path
        self.exp_name = exp_name
        self.seed = seed
        self.hyper_search = hyper_search
        self.sample_num = sample_num
        self.date = datetime.now()
        self.target_metric = target_metric
        self.root = join(root, 'experiments')
        self.pipeline_generator = None
        self.structure = None
        self.save_path = join(self.root, '{}-{}-{}'.format(self.date.year, self.date.month, self.date.day),
                              self.exp_name, 'checkpoints')

        if isinstance(self.structure, list):
            self.structure_type = 'list'
        elif isinstance(self.structure, dict):
            # self.structure_type = 'dict'
            raise ValueError("Dict structure as input parameter not implemented yet.")
        else:
            raise ValueError("Structure parameter must be a list or dict")

        self.logger = Logger(exp_name, root, self.info, self.date)
        self.start_exp = time()

        if metrics is not None:
            if isinstance(metrics, list):
                self.metrics = metrics
            else:
                raise ValueError("Metrics must be a list of strings.")
        else:
            self.metrics = ['accuracy', 'f1_macro', 'f1_weighted']

        self.logger.log['experiment_info']['metrics'] = self.metrics
        self.logger.log['experiment_info']['target_metric'] = self.target_metric

    def check_dataset(self):
        if isinstance(self.dataset, dict):
            if not ('train' in self.dataset.keys() and 'test' in self.dataset.keys()):
                raise ValueError("Input dataset must contain 'train' and 'test' keys with data.")
            elif len(self.dataset['train']) == 0 or len(self.dataset['test']) == 0:
                raise ValueError("Input dict is empty.")
        else:
            raise ValueError("Input dataset must be a dict.")
        return self

    def run(self):

        best_models = {}

        self.check_dataset()

        # analytics of dataset
        if self.data_func is not None:
            an_start = time()
            data_info = self.data_func(self.dataset)
            self.logger.log['dataset']['time'] = normal_time(time() - an_start)
            self.logger.log['dataset'].update(**data_info)

        # create PipelineGenerator
        # TODO it can be simpler
        # assert self.hyper_search in ['random', 'grid']
        # self.pipeline_generator = PipelineGenerator(self.structure, n=self.sample_num, dtype='list',
        #                                             search=self.hyper_search)

        if self.hyper_search == 'random':
            self.pipeline_generator = RandomGenerator(self.structure, n=self.sample_num)
        elif self.hyper_search == 'grid':
            self.pipeline_generator = GridGenerator(self.structure)
        else:
            raise ValueError("{} search not implemented.".format(self.hyper_search))

        # Start generating pipelines configs
        print('[ Experiment start ... ]')
        exp_start_time = time()
        for i, pipe in enumerate(self.pipeline_generator()):
            # print progress
            if i != 0:
                itime = normal_time(((time() - exp_start_time) / i) * (self.pipeline_generator.len - i))
                print('\n')
                print('[ Progress: pipe {0}/{1}; Time left: {2}; ]'.format(i+1, self.pipeline_generator.len, itime))

            self.logger.pipe_ind = i
            pipe_start = time()

            # add watcher if need
            if self.add_watcher:
                watcher = Watcher(join(self.root, '{0}-{1}-{2}'.format(self.date.year, self.date.month, self.date.day),
                                       self.exp_name), self.seed)

            dataset_i = deepcopy(self.dataset)
            for j in range(pipe.length):
                try:
                    op_start = time()
                    conf = pipe.get_op_config(j)
                    self.logger.ops[str(j)] = conf

                    if self.add_watcher:
                        test = watcher.test_config(conf, dataset_i)
                        if test is False:
                            dataset_i = pipe.step(j, dataset_i)
                            watcher.save_data(dataset_i)
                        elif test is True:
                            dataset_i = pipe.step(j, dataset_i)
                        else:
                            dataset_i = test

                    else:
                        dataset_i = pipe.step(j, dataset_i)

                    t = {'time': normal_time(time() - op_start)}
                    self.logger.ops[str(j)].update(**t)

                except:
                    print('Operation with number {0};'.format(j + 1))
                    raise

            # save best models
            self.logger.pipe_time = normal_time(time() - pipe_start)
            self.logger.pipe_res = dataset_i['results']
            self.logger.get_pipe_log()

            model = pipe.get_last_model()
            model_name = model.op_name
            if model_name not in best_models.keys():
                best_models[model_name] = 0

            if dataset_i['results'][self.target_metric] > best_models[model_name]:
                best_models[model_name] = dataset_i['results'][self.target_metric]
                if hasattr(model, 'save'):
                    model.save(join(self.save_path, model_name))
                else:
                    # only for sklearn models
                    fname = join(self.save_path, model_name)
                    if isdir(fname):
                        pass
                    else:
                        os.makedirs(fname)

                    _ = joblib.dump(model, join(fname, model_name + '.pkl'), compress=9)

        # save log
        self.logger.log['experiment_info']['full_time'] = normal_time(time() - self.start_exp)
        self.logger.save()

        # visualization of results
        path = join(self.root, '{0}-{1}-{2}'.format(self.date.year, self.date.month, self.date.day), self.exp_name)
        results_visualization(path, join(path, 'results', 'images'), self.target_metric)

        return None
