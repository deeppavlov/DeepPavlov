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
from deeppavlov.models.ner.network_ontonotes import NerNetwork
from deeppavlov.core.models.tf_model import TFModel
from deeppavlov.core.common.log import get_logger

log = get_logger(__name__)


@register('ner_ontonotes')
class NER(TFModel):
    def __init__(self, **kwargs):
        self.opt = deepcopy(kwargs)
        vocabs = self.opt.pop('vocabs', {})
        self.opt.update(vocabs)

        # Find all input parameters of the network init
        network_parameter_names = list(inspect.signature(NerNetwork.__init__).parameters)
        # Fill all provided parameters from opt
        network_parameters = {par: self.opt[par] for par in network_parameter_names if par in self.opt}

        self.sess = tf.Session()
        network_parameters['sess'] = self.sess
        self._network_parameters = network_parameters
        self._net = NerNetwork(**network_parameters)

        # Try to load the model (if there are some model files the model will be loaded from them)
        super().__init__(**kwargs)
        if self.load_path is not None:
            self.load()

    def load(self, *args, **kwargs):
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

    def train_on_batch(self, batch_x, batch_y):
        raise NotImplementedError

    @overrides
    def __call__(self, batch, *args, **kwargs):
        if isinstance(batch[0], str):
            batch = [tokenize_reg(utterance) for utterance in batch]
        return self._net.predict_on_batch(batch)

    def shutdown(self):
        self._net.shutdown()
