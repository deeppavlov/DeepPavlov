# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import inspect

from deeppavlov.core.common.registry import register
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.models.morpho_tagger.network import CharacterTagger


log = get_logger(__name__)


@register("morpho_tagger")
class MorphoTaggerWrapper(NNModel):
    """A wrapper over morphological tagger, implemented in
    :class:~deeppavlov.models.morpho_tagger.network.CharacterTagger.
    A subclass of :class:`~deeppavlov.core.models.nn_model.NNModel`

    Args:
        save_path: the path where model is saved
        load_path: the path from where model is loaded
        mode: usage mode
        **kwargs: a dictionary containing model parameters specified in the main part
            of json config that corresponds to the model
    """
    def __init__(self, save_path: str = None, load_path: str = None, mode: str = None, **kwargs):
        # Calls parent constructor. Results in creation of save_folder if it doesn't exist
        super().__init__(save_path=save_path, load_path=load_path, mode=mode)

        # Dicts are mutable! To prevent changes in config dict outside this class
        # we use deepcopy
        opt = copy.deepcopy(kwargs)

        # Finds all input parameters of the network __init__ to pass them into network later
        network_parameter_names = list(inspect.signature(CharacterTagger.__init__).parameters)
        # Fills all provided parameters from opt (opt is a dictionary formed from the model
        # json config file, except the "name" field)
        network_parameters = {par: opt[par] for par in network_parameter_names if par in opt}

        self._net = CharacterTagger(**network_parameters)

        # Finds all parameters for network train to pass them into train method later
        train_parameters_names = list(inspect.signature(self._net.train_on_batch).parameters)

        # Fills all provided parameters from opt
        train_parameters = {par: opt[par] for par in train_parameters_names if par in opt}
        self.train_parameters = train_parameters
        self.opt = opt

        # Tries to load the model from model `load_path`, if it is available
        self.load()

    def load(self):
        """Checks existence of the model file, loads the model if the file exists"""

        # General way (load path from config assumed to be the path
        # to the file including extension of the file model)
        model_file_exist = self.load_path.exists()
        path = str(self.load_path.resolve())
        # Check presence of the model files
        if model_file_exist:
            log.info('[loading model from {}]'.format(path))
            self._net.load(path)

    def save(self):
        """Saves model to the save_path, provided in config. The directory is
        already created by super().__init__, which is called in __init__ of this class"""
        path = str(self.save_path.absolute())
        log.info('[saving model to {}]'.format(path))
        self._net.save(path)

    def train_on_batch(self, *args):
        """Trains the model on a single batch.

        Args:
            *args: the list of network inputs.
            Last element of `args` is the batch of targets,
            all previous elements are training data batches
        """
        *data, labels = args
        self._net.train_on_batch(data, labels)

    def __call__(self, *x_batch, **kwargs):
        """
        Predicts answers on batch elements.

        Args:
            instance: a batch to predict answers on
        """
        return self._net.predict_on_batch(x_batch, **kwargs)

