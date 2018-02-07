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

from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.tf_backend import TfModelMeta
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.models.inferable import Inferable
from deeppavlov.models.ner.ner_network import NerNetwork as YourNetwork


@register('adapter')
class Adapter(Trainable, Inferable, metaclass=TfModelMeta):
    def __init__(self, **kwargs):
        """ Initialize the model and additional parent classes attributes

        Args:
            **kwargs: a dictionary containing parameters for model and parameters for training
                      it formed from json config file part that correspond to your model.

        """

        # Parameters for parent classes
        save_path = kwargs.get('save_path', None)
        load_path = kwargs.get('load_path', None)
        train_now = kwargs.get('train_now', None)
        mode = kwargs.get('mode', None)

        # Call parent constructors. Results in addition of attributes (save_path,
        # load_path, train_now, mode to current instance) and creation of save_folder
        # if it doesn't exist
        super().__init__(save_path=save_path, load_path=load_path,
                         train_now=train_now, mode=mode)

        # Dicts are mutable! To prevent changes in config dict outside this class
        # we use deepcopy
        opt = deepcopy(kwargs)

        # Get vocabularies. Vocabularies are made to perform token -> index / index -> token
        # transformations as well as class -> index / index -> class for classification tasks
        # TODO: use vocabs
        self.vocabs = opt.pop('vocabs', None)

        # Find all input parameters of the network __init__ to pass them into network later
        network_parameter_names = list(inspect.signature(YourNetwork.__init__).parameters)
        # Fill all provided parameters from opt (opt is a dictionary formed from the model
        # json config file, except the "name" field)
        network_parameters = {par: opt[par] for par in network_parameter_names if par in opt}

        self._net = YourNetwork(**network_parameters)

        # Find all parameters for network train to pass them into train method later
        train_parameters_names = list(inspect.signature(self._net.train_on_batch).parameters)

        # Fill all provided parameters from opt
        train_parameters = {par: opt[par] for par in train_parameters_names if par in opt}

        self.train_parameters = train_parameters

        self.opt = opt

        # Try to load the model (if there are some model files the model will be loaded from them)
        self.load()

    @overrides
    def load(self):
        """Check existence of the model file, load the model if the file exists"""

        # General way (load path from config assumed to be the path
        # to the file including extension of the file model)
        model_file_exist = self.load_path.exists()
        path = str(self.load_path.resolve())

        # TF way
        # path = str(self.load_path.resolve())
        # model_file_exist = tf.train.checkpoint_exists(path)

        # Check presence of the model files
        if model_file_exist:
            print('[loading model from {}]'.format(path), file=sys.stderr)
            self._net.load(path)

    @overrides
    def save(self):
        """Save model to the save_path, provided in config. The directory is
        already created by super().__init__ part in called in __init__ of this class"""
        path = str(self.save_path.absolute())
        print('[saving model to {}]'.format(path), file=sys.stderr)
        self._net.save(path)

    @overrides
    @check_attr_true('train_now')
    def train(self, data, *args, **kwargs):
        """ Perform training of the network given the dataset data

        Args:
            data: a dict with fields 'train', 'valid', and 'test', each field
                  contains a list of pairs (x, y), each pair is tuple, each pair
                  is a sample from the dataset. Batches are formed from the samples
            *args: not used
            **kwargs: not used

        Returns:

        """
        self._net.train(data, **self.train_parameters)

    @overrides
    def infer(self, instance):
        """Infer is similar to predict, however it should work with single samples,
        not batches."""
        return self._net.predict(instance)

    def interact(self):
        """Interactive inferrence. Type your x and get y printed"""
        s = input('Type in your x: ')
        prediction = self.infer(s)
        print(prediction)

    def shutdown(self):
        pass

    def reset(self):
        pass
