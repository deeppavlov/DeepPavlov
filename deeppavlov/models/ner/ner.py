import tensorflow as tf
from overrides import overrides
from copy import deepcopy
import inspect

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import tokenize_reg
from deeppavlov.core.common.paths import USR_PATH
from deeppavlov.models.ner.network import NerNetwork
from deeppavlov.core.models.tf_model import SimpleTFModel


@register('ner')
class NER(SimpleTFModel):
    def __init__(self, **kwargs):
        opt = deepcopy(kwargs)

        # Look for model direcory and model file names
        model_dir = opt.get('model_dir', None)
        model_file = opt.get('model_file', 'ner_model.ckpt')  # default name is ner_model
        if model_dir is None:
            model_dir = USR_PATH

        # Set these arguments to use self.model_path_ property from
        # Serializable class (inherited from Trainable)
        self._model_dir = model_dir
        self._model_file = model_file

        # Find all input parameters of the network init
        network_parameter_names = list(inspect.signature(NerNetwork.__init__).parameters)

        # Fill all provided parameters from opt
        network_parameters = {}
        for parameter_name in network_parameter_names:
            if parameter_name in opt:
                network_parameters[parameter_name] = opt[parameter_name]

        self._net = NerNetwork(**network_parameters)

        # Find all parameters for network train
        train_parameters_names = list(inspect.signature(NerNetwork.train_on_batch).parameters)
        train_parameters = {par: opt[par] for par in train_parameters_names if parameter_name in opt}
        self.train_parameters = train_parameters

        # Try to load the model (if there are some model files the model will be loaded from them)
        self.load()

    @overrides
    def load(self):
        # Check prescence of the model files
        path = str(self.model_path_.absolute())
        if tf.train.checkpoint_exists(path):
            self._net.load(path)

    @overrides
    def save(self):
        self.model_path_.parent.mkdir(parents=True, exist_ok=True)
        path = str(self.model_path_.absolute())
        print('[saving model to {}]'.format(path))

        self._net.save(path)

    @overrides
    def train(self, data, default_n_epochs=10):
        if self.train_now:
            print('Training NER network')
            epochs = self.train_parameters.get('epochs', default_n_epochs)
            for epoch in range(epochs):
                self._net.train(data, **self.train_parameters)
                self._net.eval_conll(data.iter_all('valid'), short_report=False, data_type='valid')
            self._net.eval_conll(data.iter_all('train'), short_report=False, data_type='train')
            self._net.eval_conll(data.iter_all('test'), short_report=False, data_type='test')
            self.save()
        else:
            print('Loading NER network')
            self._net.load()

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
