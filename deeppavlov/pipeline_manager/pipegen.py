import numpy as np

from itertools import product
from copy import deepcopy

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.log import get_logger
from deeppavlov.pipeline_manager.hyperpar import HyperPar


log = get_logger(__name__)


class PipeGen:
    def __init__(self, config_path: str, n=10, stype: str ='grid') -> object:
        self.config_path = config_path
        self.N = n
        self.stype = stype
        self.pipes = []
        self.main_config = None
        self.chainer = None
        self.structure = None
        self.get_structure()

        if self.stype not in ['grid', 'random']:
            raise ValueError("Sorry {0} search not implemented."
                             " At the moment you can use only 'random' and 'grid' search.".format(self.stype))
        elif self.stype == 'random':
            self.len = 0
            self.random_get_len()
        elif self.stype == 'grid':
            self.len = 1
            self.grid_get_len()

        self.generator = self.pipeline_gen()

    def __call__(self, *args, **kwargs):
        return self.generator

    def get_structure(self):
        self.main_config = read_json(self.config_path)
        if 'chainer' not in self.main_config:
            raise ConfigError("Main config file not contain 'chainer' component."
                              "Structure search can not be started without this component.")

        self.chainer = self.main_config.pop('chainer')
        self.structure = self.chainer['pipe']

        return self

    def pipeline_gen(self):
        if self.stype == 'random':
            pipe_gen = self.random_conf_gen()
        elif self.stype == 'grid':
            pipe_gen = self.grid_conf_gen()
        else:
            raise ValueError("Sorry {0} search not implemented."
                             " At the moment you can use only 'random' and 'grid' search.".format(self.stype))

        for pipe in pipe_gen:
            new_config = deepcopy(self.main_config)
            new_config['chainer'] = deepcopy(self.chainer)
            chainer_pipe = list(pipe)
            new_config['chainer']['pipe'] = chainer_pipe
            yield new_config

    def random_get_len(self):
        test = []
        lst = []

        for x in self.structure:
            ln = []
            for y in x:
                if "search" not in y.keys():
                    ln.append(False)
                else:
                    ln.append(True)
            test.append(ln)

        zgen = product(*test)
        for x in zgen:
            lst.append(x)

        ks = 0
        k = 0
        for x in lst:
            if True not in x:
                k += 1
            else:
                ks += 1

        self.len = k + ks * self.N

        del test, lst, zgen

        return self

    @staticmethod
    def get_p(z):
        if 'search' in z.keys():
            l_ = list()
            for key, it in z.items():
                if key == 'search':
                    pass
                else:
                    if isinstance(it, list):
                        l_.append(len(it))
                    else:
                        pass
            p = 1
            for q in l_:
                p *= q
            return p
        else:
            return 1

    @staticmethod
    def grid_param_gen(conf):
        search_conf = deepcopy(conf)
        list_of_var = []

        # delete "search" key and element
        del search_conf['search']

        values = list()
        keys = list()

        static_keys = list()
        static_values = list()
        for key, item in search_conf.items():
            if isinstance(search_conf[key], list):
                values.append(item)
                keys.append(key)
            elif isinstance(search_conf[key], dict):
                raise ValueError("Grid search are not supported 'dict', that contain values of parameters.")
            elif isinstance(search_conf[key], tuple):
                raise ValueError("Grid search are not supported 'tuple', that contain values of parameters.")
            else:
                static_values.append(search_conf[key])
                static_keys.append(key)

        valgen = product(*values)

        config = {}
        for i in range(len(static_keys)):
            config[static_keys[i]] = static_values[i]

        for val in valgen:
            cop = deepcopy(config)
            for i, v in enumerate(val):
                cop[keys[i]] = v
            list_of_var.append(cop)

        return list_of_var

    def grid_get_len(self):
        leng = []
        for x in self.structure:
            k = 0
            for y in x:
                k += self.get_p(y)
            leng.append(k)

        for x in leng:
            self.len *= x

        return self

    # random generation
    def random_conf_gen(self):
        for x in self.structure:
            self.pipes.append(x)

        lgen = product(*self.pipes)
        for pipe in lgen:
            search = False
            pipe = list(pipe)

            for conf in pipe:
                if "search" in conf.keys():
                    search = True
                    break

            if search:
                ops_samples = {}
                for i, conf in enumerate(pipe):
                    if "search" in conf.keys():
                        search_conf = deepcopy(conf)
                        del search_conf['search']

                        sample_gen = HyperPar(**search_conf)
                        ops_samples[str(i)] = list()
                        for j in range(self.N):
                            conf_j = sample_gen.sample_params()
                            # fix dtype for json dump
                            for key in conf_j.keys():
                                if isinstance(conf_j[key], np.int64):
                                    conf_j[key] = int(conf_j[key])

                            ops_samples[str(i)].append(conf_j)

                for i in range(self.N):
                    for key, item in ops_samples.items():
                        pipe[int(key)] = item[i]
                        yield pipe
            else:
                yield pipe

    # grid generation
    def grid_conf_gen(self):

        def update(el):
            lst = []
            if 'search' not in el.keys():
                lst.append(el)
            else:
                lst.extend(self.grid_param_gen(el))
            return lst

        for i, x in enumerate(self.structure):
            ln = []
            for y in x:
                ln.extend(update(y))
            self.pipes.append(ln)

        return product(*self.pipes)
