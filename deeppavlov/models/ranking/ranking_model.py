from overrides import overrides
from copy import deepcopy
import inspect
import sys
from functools import reduce
import operator
import numpy as np

from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.models.ranking.ranking_network import RankingNetwork
from deeppavlov.models.ranking.dict import InsuranceDict
from deeppavlov.models.ranking.emb_dict import EmbeddingsDict


@register('ranking_model')
class RankingModel(NNModel):
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
        self.vocabs = opt.get('vocabs', None)

        self.dict = InsuranceDict(opt["vocabs_path"])

        embdict_parameter_names = list(inspect.signature(EmbeddingsDict.__init__).parameters)
        embdict_parameters = {par: opt[par] for par in embdict_parameter_names if par in opt}
        self.embdict= EmbeddingsDict(self.dict.toks, **embdict_parameters)

        # Find all input parameters of the network __init__ to pass them into network later
        network_parameter_names = list(inspect.signature(RankingNetwork.__init__).parameters)
        # Fill all provided parameters from opt (opt is a dictionary formed from the model
        # json config file, except the "name" field)
        network_parameters = {par: opt[par] for par in network_parameter_names if par in opt}

        self._net = RankingNetwork(self.embdict.emb_matrix, **network_parameters)

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

        # Check presence of the model files
        if model_file_exist:
            print('[loading model from {}]'.format(path), file=sys.stderr)
            self._net.load(path+'.h5')
            self.dict.load(path+'.npy')

    @overrides
    def save(self):
        """Save model to the save_path, provided in config. The directory is
        already created by super().__init__ part in called in __init__ of this class"""
        path = str(self.save_path.absolute())
        print('[saving model to {}]'.format(path), file=sys.stderr)
        self._net.save(path+'.h5')
        self.dict.save(path+'.npy')

    @check_attr_true('train_now')
    def train_on_batch(self, x, y):

        if self.dict.label2emb_vocab[0] is not None:
            for i in range(len(self.dict.label2emb_vocab)):
                self.dict.label2emb_vocab[i] = None

        context, response, negative_response = x
        c = self.dict.make_toks(context, type="context")
        c = self.embdict.make_ints(c)
        rp = self.dict.make_toks(response, type="response")
        rp = self.embdict.make_ints(rp)
        rn = self.dict.make_toks(negative_response, type="response")
        rn = self.embdict.make_ints(rn)
        b = [c, rp, rn], y
        self._net.train_on_batch(b)

    @overrides
    def __call__(self, batch):

        if self.dict.label2emb_vocab[0] is None:
            r = []
            for i in range(len(self.dict.label2toks_vocab)):
                r.append(self.dict.label2toks_vocab[i])
            r = self.embdict.make_ints(r)
            response_embeddings = self._net.predict_response_emb([r, r, r], 512)
            for i in range(len(self.dict.label2toks_vocab)):
                self.dict.label2emb_vocab[i] = response_embeddings[i]

        context = [el[0] for el in batch]
        response = [el[1] for el in batch]
        batch_size = len(response)
        ranking_length = len(response[0])
        # context = [ranking_length * [el] for el in context]
        # context = [reduce(operator.concat, context)][0]
        response = reduce(operator.concat, response)
        response = [response[i:batch_size*ranking_length:ranking_length] for i in range(ranking_length)]
        y_pred = []
        for i in range(ranking_length):
            c = self.dict.make_toks(context, type="context")
            c = self.embdict.make_ints(c)
            r = response[i]
            r = self.dict.make_toks(r, type="response")
            r = self.embdict.make_ints(r)
            b = [c, r, r]
            yp = self._net.predict_on_batch(b)
            y_pred.append(yp)
        y_pred = np.hstack(y_pred)
        return y_pred

    def interact(self):
        """Interactive inferrence. Type your x and get y printed"""
        s = input('Type in your x: ')

        prediction = self.infer(s)
        print(prediction)

    def shutdown(self):
        pass

    def reset(self):
        pass
