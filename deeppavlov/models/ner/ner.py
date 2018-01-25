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
import sys

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import tokenize_reg
from deeppavlov.models.ner.network import NerNetwork
from deeppavlov.core.models.tf_model import SimpleTFModel


@register('ner')
class NER(SimpleTFModel):
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        opt = deepcopy(kwargs)
        vocabs = opt.pop('vocabs')
        opt.update(vocabs)

        # Find all input parameters of the network init
        network_parameter_names = list(inspect.signature(NerNetwork.__init__).parameters)
        # Fill all provided parameters from opt
        network_parameters = {par: opt[par] for par in network_parameter_names if par in opt}

        self._net = NerNetwork(**network_parameters)

        # Find all parameters for network train
        train_parameters_names = list(inspect.signature(NerNetwork.train_on_batch).parameters)
        # Fill all provided parameters from opt
        train_parameters = {par: opt[par] for par in train_parameters_names if par in opt}
        self.train_parameters = train_parameters

        self.opt = opt

        # Try to load the model (if there are some model files the model will be loaded from them)
        self.load()

    @overrides
    def load(self):
        path = str(self.ser_path.absolute())
        # Check presence of the model files
        if tf.train.checkpoint_exists(path):
            print('[loading model from {}]'.format(path), file=sys.stderr)
            self._net.load(path)

    @overrides
    def save(self):
        self.ser_path.parent.mkdir(parents=True, exist_ok=True)
        path = str(self.ser_path.absolute())
        print('[saving model to {}]'.format(path), file=sys.stderr)
        self._net.save(path)

    @overrides
    def train(self, data, default_n_epochs=5):
        if self.train_now:
            print('Training NER network', file=sys.stderr)
            epochs = self.opt.get('epochs', default_n_epochs)
            for epoch in range(epochs):
                self._net.train(data, **self.train_parameters)
                self._net.eval_conll(data.iter_all('valid'), short_report=False, data_type='valid')
            self._net.eval_conll(data.iter_all('train'), short_report=False, data_type='train')
            self._net.eval_conll(data.iter_all('test'), short_report=False, data_type='test')
            self.save()
        else:
            print('Loading NER network', file=sys.stderr)
            self._net.load()

    def train_on_batch(self, batch):
        self._net.train_on_batch(batch, **self.train_parameters)


    @overrides
    def infer(self, sample, *args, **kwargs):
        sample = sample.strip()
        if not len(sample):
            return ''
        return self._net.predict_on_batch([self.preprocess_tokenize(sample)])[0]

    def preprocess_tokenize(self, utterance):
        sample = tokenize_reg(utterance)
        return sample

    def interact(self):
        s = input('Type in the message you want to tag: ')
        prediction = self.infer(s)
        print(prediction)

    def shutdown(self):
        self.ner.shutdown()

    def reset(self):
        pass
