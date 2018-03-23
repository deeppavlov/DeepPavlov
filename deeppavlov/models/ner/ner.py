"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import tensorflow as tf
from overrides import overrides
from copy import deepcopy
import inspect
import json

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import tokenize_reg
from deeppavlov.models.ner.network import NerNetwork
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)


@register('ner')
class NER(TFModel):
    GRAPH_PARAMS = ["n_filters",
                    "filter_width",
                    "token_embeddings_dim",
                    "char_embeddings_dim",
                    "use_char_embeddings",
                    "use_batch_norm",
                    "use_crf",
                    "net_type",
                    "char_filter_width",
                    "cell_type"]
    VOCABS = ['word_vocab',
              'char_vocab',
              'tag_vocab']

    def __init__(self, **kwargs):
        self.opt = deepcopy(kwargs)
        vocabs = self.opt.pop('vocabs')
        self.opt.update(vocabs)

        # Find all input parameters of the network init
        network_parameter_names = list(inspect.signature(NerNetwork.__init__).parameters)
        # Fill all provided parameters from opt
        network_parameters = {par: self.opt[par] for par in network_parameter_names if par in self.opt}

        self.sess = tf.Session()
        network_parameters['sess'] = self.sess
        self._network_parameters = network_parameters
        self._net = NerNetwork(**network_parameters)

        # Find all parameters for network train
        train_parameters_names = list(inspect.signature(NerNetwork.train_on_batch).parameters)
        # Fill all provided parameters from opt
        train_parameters = {par: self.opt[par] for par in train_parameters_names if par in self.opt}
        self.train_parameters = train_parameters

        # Try to load the model (if there are some model files the model will be loaded from them)
        super().__init__(**kwargs)
        if self.load_path is not None:
            self.load()

    def load(self, *args, **kwargs):
        self.load_params()
        super().load(*args, **kwargs)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        self.save_params()

    def save_params(self):
        params_to_save = {param: self.opt.get(param, None) for param in self.GRAPH_PARAMS}
        for vocab in self.VOCABS:
            params_to_save[vocab] = [self.opt[vocab][i] for i in range(len(self.opt[vocab]))]
        path = str(self.save_path.with_suffix('.json').resolve())
        log.info('[saving parameters to {}]'.format(path))
        with open(path, 'w') as fp:
            json.dump(params_to_save, fp, indent=4)

    def load_params(self):
        path = self.load_path.with_suffix('.json').resolve()
        if not path.exists():
            return
        log.info('[loading parameters from {}]'.format(path))
        with open(path, 'r') as fp:
            params = json.load(fp)
        for p in self.GRAPH_PARAMS:
            if self.opt.get(p, None) != params[p]:
                raise ValueError("`{}` parameter must be equal to "
                                 "saved model parameter value `{}`" \
                                 .format(p, params[p]))
        for vocab_name in self.VOCABS:
            vocab = [self.opt[vocab_name][i] for i in range(len(self.opt[vocab_name]))]
            if vocab != params[vocab_name]:
                raise ValueError("`{}` vocabulary must be equal in created and "
                                 "saved model".format(vocab_name))

    def train_on_batch(self, batch_x, batch_y):
        self._net.train_on_batch(batch_x, batch_y, **self.train_parameters)

    @overrides
    def __call__(self, batch, *args, **kwargs):
        if isinstance(batch[0], str):
            batch = [tokenize_reg(utterance) for utterance in batch]
        return self._net.predict_on_batch(batch)

    def shutdown(self):
        self._net.shutdown()
