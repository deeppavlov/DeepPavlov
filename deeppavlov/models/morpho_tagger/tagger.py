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
import sys
import copy
import inspect

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.models.morpho_tagger.network import CharacterTagger


log = get_logger(__name__)


@register("morpho_tagger")
class MorphoTaggerWrapper(NNModel):

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
        opt = copy.deepcopy(kwargs)

        # Find all input parameters of the network __init__ to pass them into network later
        network_parameter_names = list(inspect.signature(CharacterTagger.__init__).parameters)
        # Fill all provided parameters from opt (opt is a dictionary formed from the model
        # json config file, except the "name" field)
        network_parameters = {par: opt[par] for par in network_parameter_names if par in opt}

        self._net = CharacterTagger(**network_parameters)

        # Find all parameters for network train to pass them into train method later
        train_parameters_names = list(inspect.signature(self._net.train_on_batch).parameters)

        # Fill all provided parameters from opt
        train_parameters = {par: opt[par] for par in train_parameters_names if par in opt}

        self.train_parameters = train_parameters

        self.opt = opt

        # Try to load the model (if there are some model files the model will be loaded from them)
        self.load()

    def load(self):
        """Check existence of the model file, load the model if the file exists"""

        # General way (load path from config assumed to be the path
        # to the file including extension of the file model)
        model_file_exist = self.load_path.exists()
        path = str(self.load_path.resolve())
        # Check presence of the model files
        if model_file_exist:
            log.info('[loading model from {}]'.format(path))
            self._net.load(path)

    def save(self):
        """Save model to the save_path, provided in config. The directory is
        already created by super().__init__ part in called in __init__ of this class"""
        path = str(self.save_path.absolute())
        log.info('[saving model to {}]'.format(path))
        self._net.save(path)

    def train_on_batch(self, *args):
        """ Perform training of the network given the dataset data

        Args:
            x: an x batch
            y: an y batch

        Returns:

        """
        if len(args) > 2:
            data, labels = [list(x) for x in args[:-1]], list(args[-1])
        else:
            data, labels = args
        self._net.train_on_batch(data, labels, **self.train_parameters)

    def __call__(self, *x_batch, **kwargs):
        """
        Predicts answers on batch elements.

        Args:
            instance: a batch to predict answers on

        """
        # if len(args) > 0:
        #     x_batch = [x_batch] + list(args)
        return self._net.predict_on_batch(x_batch, **kwargs)

