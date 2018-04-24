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

from overrides import overrides
from copy import deepcopy
import inspect
from functools import reduce
import operator
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.models.ranking.ranking_network import RankingNetwork
from deeppavlov.models.ranking.dict import InsuranceDict
from deeppavlov.models.ranking.emb_dict import Embeddings
from deeppavlov.core.common.log import get_logger


log = get_logger(__name__)


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

        opt = deepcopy(kwargs)
        self.train_now = opt['train_now']
        self.opt = opt
        self.interact_pred_num = opt['interact_pred_num']
        self.vocabs = opt.get('vocabs', None)

        dict_parameter_names = list(inspect.signature(InsuranceDict.__init__).parameters)
        dict_parameters = {par: opt[par] for par in dict_parameter_names if par in opt}
        network_parameter_names = list(inspect.signature(RankingNetwork.__init__).parameters)
        self.network_parameters = {par: opt[par] for par in network_parameter_names if par in opt}

        self.dict = InsuranceDict(**dict_parameters)

        self.load()

        train_parameters_names = list(inspect.signature(self._net.train_on_batch).parameters)
        self.train_parameters = {par: opt[par] for par in train_parameters_names if par in opt}

    @overrides
    def load(self):
        if not self.load_path.exists():
            log.info("[initializing new `{}`]".format(self.__class__.__name__))
            self.dict.init_from_scratch()
            self._net = RankingNetwork(len(self.dict.tok2int_vocab), **self.network_parameters)
            embdict_parameter_names = list(inspect.signature(Embeddings.__init__).parameters)
            embdict_parameters = {par: self.opt[par] for par in embdict_parameter_names if par in self.opt}
            embdict= Embeddings(self.dict.tok2int_vocab, **embdict_parameters)
            self._net.set_emb_matrix(embdict.emb_matrix)
        else:
            log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
            self.dict.load()
            self._net = RankingNetwork(len(self.dict.tok2int_vocab), **self.network_parameters)
            self._net.load(self.load_path)

    @overrides
    def save(self):
        """Save model to the save_path, provided in config. The directory is
        already created by super().__init__ part in called in __init__ of this class"""
        log.info('[saving model to {}]'.format(self.save_path.resolve()))
        self._net.save(self.save_path)
        self.set_embeddings()
        self.dict.save()

    @check_attr_true('train_now')
    def train_on_batch(self, x, y):
        self.reset_embeddings()
        context = [el[0] for el in x]
        response = [el[1] for el in x]
        negative_response = [el[2] for el in x]
        c = self.dict.make_toks(context, type="context")
        c = self.dict.make_ints(c)
        rp = self.dict.make_toks(response, type="response")
        rp = self.dict.make_ints(rp)
        rn = self.dict.make_toks(negative_response, type="response")
        rn = self.dict.make_ints(rn)
        b = [c, rp, rn], y
        self._net.train_on_batch(b)

    @overrides
    def __call__(self, batch):
        self.set_embeddings()
        if type(batch[0]) == list:
            context = [el[0] for el in batch]
            c = self.dict.make_toks(context, type="context")
            c = self.dict.make_ints(c)
            c_emb = self._net.predict_context_emb([c, c, c], bs=len(batch))
            response = [el[1] for el in batch]
            batch_size = len(response)
            ranking_length = len(response[0])
            response = reduce(operator.concat, response)
            response = [response[i:batch_size*ranking_length:ranking_length] for i in range(ranking_length)]
            y_pred = []
            for i in range(ranking_length):
                r_emb = [self.dict.response2emb_vocab[el] for el in response[i]]
                r_emb = np.vstack(r_emb)
                yp = np.sum(c_emb * r_emb, axis=1) / np.linalg.norm(c_emb, axis=1) / np.linalg.norm(r_emb, axis=1)
                y_pred.append(np.expand_dims(yp, axis=1))
            y_pred = np.hstack(y_pred)
            return y_pred

        elif type(batch[0]) == str:
            c_input = tokenize(batch)
            c_input = self.dict.make_ints(c_input)
            c_input_emb = self._net.predict_context_emb([c_input, c_input, c_input], bs=1)

            c_emb = [self.dict.context2emb_vocab[i] for i in range(len(self.dict.context2emb_vocab))]
            c_emb = np.vstack(c_emb)
            pred_cont = np.sum(c_input_emb * c_emb, axis=1)\
                     / np.linalg.norm(c_input_emb, axis=1) / np.linalg.norm(c_emb, axis=1)
            pred_cont = np.flip(np.argsort(pred_cont), 0)[:self.interact_pred_num]
            pred_cont = [' '.join(self.dict.context2toks_vocab[el]) for el in pred_cont]

            r_emb = [self.dict.response2emb_vocab[i] for i in range(len(self.dict.response2emb_vocab))]
            r_emb = np.vstack(r_emb)
            pred_resp = np.sum(c_input_emb * r_emb, axis=1)\
                     / np.linalg.norm(c_input_emb, axis=1) / np.linalg.norm(r_emb, axis=1)
            pred_resp = np.flip(np.argsort(pred_resp), 0)[:self.interact_pred_num]
            pred_resp = [' '.join(self.dict.response2toks_vocab[el]) for el in pred_resp]
            y_pred = [{"contexts": pred_cont, "responses": pred_resp}]
            return y_pred

    def set_embeddings(self):
        if self.dict.response2emb_vocab[0] is None:
            r = []
            for i in range(len(self.dict.response2toks_vocab)):
                r.append(self.dict.response2toks_vocab[i])
            r = self.dict.make_ints(r)
            response_embeddings = self._net.predict_response_emb([r, r, r], 512)
            for i in range(len(self.dict.response2toks_vocab)):
                self.dict.response2emb_vocab[i] = response_embeddings[i]
        if self.dict.context2emb_vocab[0] is None:
            contexts = []
            for i in range(len(self.dict.context2toks_vocab)):
                contexts.append(self.dict.context2toks_vocab[i])
            contexts = self.dict.make_ints(contexts)
            context_embeddings = self._net.predict_context_emb([contexts, contexts, contexts], 512)
            for i in range(len(self.dict.context2toks_vocab)):
                self.dict.context2emb_vocab[i] = context_embeddings[i]

    def reset_embeddings(self):
        if self.dict.response2emb_vocab[0] is not None:
            for i in range(len(self.dict.response2emb_vocab)):
                self.dict.response2emb_vocab[i] = None
        if self.dict.context2emb_vocab[0] is not None:
            for i in range(len(self.dict.context2emb_vocab)):
                self.dict.context2emb_vocab[i] = None

    def shutdown(self):
        pass

    def reset(self):
        pass


def tokenize(sen_list):
    sen_tokens_list = []
    for sen in sen_list:
        sent_toks = sent_tokenize(sen)
        word_toks = [word_tokenize(el) for el in sent_toks]
        tokens = [val for sublist in word_toks for val in sublist]
        tokens = [el for el in tokens if el != '']
        sen_tokens_list.append(tokens)
    return sen_tokens_list
