import numpy as np

from os.path import join
from itertools import product
from copy import deepcopy

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.log import get_logger
from deeppavlov.pipeline_manager.utils import HyperPar


log = get_logger(__name__)


class PipeGen:
    """
    The class implements the generator of standard DeepPavlov configs.
    """
    def __init__(self, config_path: str, save_path: str, stype: str ='grid', n=10, test_mode=False):
        """
        Initialize generator with input params.

        Args:
            config_path: str; path to config file with search pattern.
            save_path: str; path to folder with pipelines checkpoints
            stype: str; random or grid - the trigger that determines type of hypersearch
            n: int; determines the number of generated pipelines, if hyper_search == random.
        """
        self.test_mode = test_mode
        self.save_path = save_path
        self.config_path = config_path
        self.N = n
        self.stype = stype
        self.pipes = []
        self.main_config = None
        self.chainer = None
        self.structure = None
        self.get_structure()
        self._check_component_name()

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

    def _check_component_name(self) -> None:
        for i, component in enumerate(self.structure):
            for j, example in enumerate(component):
                if example is not None:
                    if "component_name" not in example.keys():
                        raise ConfigError("The pipeline element in config file, on position {0} and with number {1}"
                                          "don't contain the 'component_name' key.".format(i+1, j+1))
        return None

    def get_structure(self):
        """
        Read search pattern from config (json) file.

        Returns:
            self
        """
        self.main_config = read_json(self.config_path)
        if 'chainer' not in self.main_config:
            raise ConfigError("Main config file not contain 'chainer' component."
                              "Structure search can not be started without this component.")

        self.chainer = self.main_config.pop('chainer')
        self.structure = self.chainer['pipe']

        return self

    def random_get_len(self):
        """
        Computes number of generated pipelines.
        Returns:
            self
        """
        test = []
        lst = []

        for x in self.structure:
            ln = []
            for y in x:
                if y is None:
                    ln.append(False)
                else:
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
        if z is None:
            return 1
        else:
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

    def grid_get_len(self):
        """
        Computes number of generated pipelines.
        Returns:
            self
        """
        leng = []
        for x in self.structure:
            k = 0
            for y in x:
                k += self.get_p(y)
            leng.append(k)

        for x in leng:
            self.len *= x

        return self

    def pipeline_gen(self):
        """
        Generate DeepPavlov standard configs (dicts).
        Returns:
            python generator
        """
        if self.stype == 'random':
            pipe_gen = self.random_conf_gen()
        elif self.stype == 'grid':
            pipe_gen = self.grid_conf_gen()
        else:
            raise ValueError("Sorry {0} search not implemented."
                             " At the moment you can use only 'random' and 'grid' search.".format(self.stype))

        for k, pipe in enumerate(pipe_gen):
            new_config = deepcopy(self.main_config)
            new_config['chainer'] = deepcopy(self.chainer)
            chainer_pipe = list(pipe)
            chainer_pipe = self.change_load_path(chainer_pipe, k)
            new_config['chainer']['pipe'] = chainer_pipe
            yield new_config

    # random generation
    def random_conf_gen(self):
        """
        Creates generator that return all possible pipelines.
        Returns:
            python generator
        """
        for x in self.structure:
            self.pipes.append(x)

        lgen = product(*self.pipes)
        for pipe in lgen:
            search = False
            pipe = list(pipe)

            for conf in pipe:
                if conf is None:
                    pipe.remove(conf)
                else:
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

    @staticmethod
    def grid_param_gen(conf):
        """
        Compute cartesian product of config parameters.
        Args:
            conf: dict; component of search pattern

        Returns:
            list
        """
        search_conf = deepcopy(conf)
        list_of_var = []

        # delete "search" key and element
        del search_conf['search']

        values = list()
        keys = list()

        static_keys = list()
        static_values = list()
        stop_keys = ['in', 'in_x', 'in_y', 'out']
        for key, item in search_conf.items():
            if key not in stop_keys:
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

    # grid generation
    def grid_conf_gen(self):
        """
        Creates generator that return all possible pipelines.
        Returns:
            python generator
        """
        def update(el):
            lst = []
            if el is not None:
                if 'search' not in el.keys():
                    lst.append(el)
                else:
                    lst.extend(self.grid_param_gen(el))
            else:
                lst.append(el)
            return lst

        for i, x in enumerate(self.structure):
            ln = []
            for y in x:
                ln.extend(update(y))
            self.pipes.append(ln)

        lgen = product(*self.pipes)
        for pipe in lgen:
            pipe = list(pipe)
            for conf in pipe:
                if conf is None:
                    pipe.remove(conf)
            yield pipe

    def change_load_path(self, config, n):
        """
        Change save_path and load_path attributes in standard config.
        Args:
            config: dict; the chainer content.
            n: int; pipeline number

        Returns:
            config: dict; new config with changed save and load paths
        """
        for component in config:
            if component.get('main') is True:
                if component.get('save_path', None) is not None:
                    sp = component['save_path'].split('/')[-1]
                    if not self.test_mode:
                        component['save_path'] = join('..', self.save_path, 'pipe_{}'.format(n+1), sp)
                    else:
                        component['save_path'] = join('..', self.save_path, "tmp", 'pipe_{}'.format(n + 1), sp)
                if component.get('load_path', None) is not None:
                    lp = component['load_path'].split('/')[-1]
                    if not self.test_mode:
                        component['load_path'] = join('..', self.save_path, 'pipe_{}'.format(n+1), lp)
                    else:
                        component['load_path'] = join('..', self.save_path, "tmp", 'pipe_{}'.format(n + 1), lp)
            else:
                if component.get('save_path', None) is not None:
                    sp = component['save_path'].split('/')[-1]
                    if not self.test_mode:
                        component['save_path'] = join('..', self.save_path, sp)
                    else:
                        component['save_path'] = join('..', self.save_path, "tmp", sp)
                # if component.get('load_path', None) is not None:
                #     lp = component['load_path'].split('/')[-1]
                #     if not self.test_mode:
                #         component['load_path'] = join('..', self.save_path, lp)
                #     else:
                #         component['load_path'] = join('..', self.save_path, "tmp", lp)
        return config

    def __call__(self, *args, **kwargs):
        return self.generator
