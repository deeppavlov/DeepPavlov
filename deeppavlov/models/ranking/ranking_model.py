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
import random
from nltk.tokenize import sent_tokenize, word_tokenize

from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.nn_model import NNModel
from deeppavlov.models.ranking.ranking_network import RankingNetwork
from deeppavlov.models.ranking.insurance_dict import InsuranceDict
from deeppavlov.models.ranking.sber_faq_dict import SberFAQDict
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
        self.hard_triplets = opt.get('hard_triplets')
        self.hardest_positives = opt.get('hardest_positives')
        self.semi_hard = opt.get('semi_hard')
        self.num_hardest_samples = opt.get('num_hardest_samples')
        self.update_embeddings = opt.get('update_embeddings', 'on_validation')
        self.opt = opt
        self.interact_pred_num = opt['interact_pred_num']
        self.vocabs = opt.get('vocabs', None)

        if self.opt["vocab_name"] == "insurance":
            dict_parameter_names = list(inspect.signature(InsuranceDict.__init__).parameters)
            dict_parameters = {par: opt[par] for par in dict_parameter_names if par in opt}
            self.dict = InsuranceDict(**dict_parameters)
        if self.opt["vocab_name"] == "sber_faq":
            dict_parameter_names = list(inspect.signature(SberFAQDict.__init__).parameters)
            dict_parameters = {par: opt[par] for par in dict_parameter_names if par in opt}
            self.dict = SberFAQDict(**dict_parameters)

        embdict_parameter_names = list(inspect.signature(Embeddings.__init__).parameters)
        embdict_parameters = {par: self.opt[par] for par in embdict_parameter_names if par in self.opt}
        self.embdict= Embeddings(**embdict_parameters)


        # self.dict: DictInterface = kwargs['vocab']

        network_parameter_names = list(inspect.signature(RankingNetwork.__init__).parameters)
        self.network_parameters = {par: opt[par] for par in network_parameter_names if par in opt}

        self.load()

        train_parameters_names = list(inspect.signature(self._net.train_on_batch).parameters)
        self.train_parameters = {par: opt[par] for par in train_parameters_names if par in opt}

    @overrides
    def load(self):
        if not self.load_path.exists():
            log.info("[initializing new `{}`]".format(self.__class__.__name__))
            self.dict.init_from_scratch()
            self.embdict.init_from_scratch(self.dict.tok2int_vocab)
            self._net = RankingNetwork(toks_num=len(self.dict.tok2int_vocab),
                                       emb_dict=self.embdict,
                                       **self.network_parameters)
            self._net.init_from_scratch(self.embdict.emb_matrix)
        else:
            log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
            self.dict.load()
            self.embdict.load()
            self._net = RankingNetwork(toks_num=len(self.dict.tok2int_vocab),
                                       emb_dict=self.embdict,
                                       **self.network_parameters)
            self._net.load(self.load_path)

    @overrides
    def save(self):
        """Save model to the save_path, provided in config. The directory is
        already created by super().__init__ part in called in __init__ of this class"""
        log.info('[saving model to {}]'.format(self.save_path.resolve()))
        self._net.save(self.save_path)
        if self.update_embeddings == 'on_validation':
            self.set_embeddings()
        self.dict.save()
        self.embdict.save()

    @check_attr_true('train_now')
    def train_on_batch(self, x, y):
        if self.update_embeddings == 'on_validation':
            self.reset_embeddings()
        if self.hard_triplets:
            c, rp, rn = self.make_hard_triplets(x, self._net)
            y = np.ones((len(c), len(x[0][1])))
        else:
            context = [el[0] for el in x]
            pos_neg_response = [el[1] for el in x]
            response = [el[0] for el in pos_neg_response]
            negative_response = [el[1] for el in pos_neg_response]
            c = self.dict.make_toks(context, type="context")
            c = self.dict.make_ints(c)
            rp = self.dict.make_toks(response, type="response")
            rp = self.dict.make_ints(rp)
            rn = self.dict.make_toks(negative_response, type="response")
            rn = self.dict.make_ints(rn)
        b = [c, rp, rn], np.asarray(y)
        self._net.train_on_batch(b)

    def make_hard_triplets(self, x, net):
        samples = [el[2] for el in x]
        labels = np.array([el[3] for el in x])
        batch_size = len(samples)
        num_samples = len(samples[0])
        samp = [y for el in samples for y in el]
        s = self.dict.make_toks(samp, type="context")
        s = self.dict.make_ints(s)

        embeddings = net.predict_context([s, s, s], 512)
        embeddings = embeddings / np.expand_dims(np.linalg.norm(embeddings, axis=1), axis=1)
        dot_product = embeddings @ embeddings.T
        square_norm = np.diag(dot_product)
        distances = np.expand_dims(square_norm, 0) - 2.0 * dot_product + np.expand_dims(square_norm, 1)
        distances = np.maximum(distances, 0.0)
        distances = np.sqrt(distances)

        mask_anchor_negative = np.expand_dims(np.repeat(labels, num_samples), 0)\
                               != np.expand_dims(np.repeat(labels, num_samples), 1)
        mask_anchor_negative = mask_anchor_negative.astype(float)
        max_anchor_negative_dist = np.max(distances, axis=1, keepdims=True)
        anchor_negative_dist = distances + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        if self.num_hardest_samples is not None:
            hard = np.argsort(anchor_negative_dist, axis=1)[:, :self.num_hardest_samples]
            ind = np.random.randint(self.num_hardest_samples, size=batch_size * num_samples)
            hardest_negative_ind = hard[batch_size * num_samples * [True], ind]
        else:
            hardest_negative_ind = np.argmin(anchor_negative_dist, axis=1)

        mask_anchor_positive = np.expand_dims(np.repeat(labels, num_samples), 0) \
                               == np.expand_dims(np.repeat(labels, num_samples), 1)
        mask_anchor_positive = mask_anchor_positive.astype(float)
        anchor_positive_dist = mask_anchor_positive * distances

        c =[]
        rp = []
        rn = []
        hrds = []

        if self.hardest_positives:

            if self.semi_hard:
                hardest_positive_ind = []
                hardest_negative_ind = []
                for p, n in zip(anchor_positive_dist, anchor_negative_dist):
                    no_samples = True
                    p_li = list(zip(p, np.arange(batch_size * num_samples), batch_size * num_samples * [True]))
                    n_li = list(zip(n, np.arange(batch_size * num_samples), batch_size * num_samples * [False]))
                    pn_li = sorted(p_li + n_li, key=lambda el: el[0])
                    for i, x in enumerate(pn_li):
                        if not x[2]:
                            for y in pn_li[:i][::-1]:
                                if y[2] and y[0] > 0.0:
                                    assert (x[1] != y[1])
                                    hardest_negative_ind.append(x[1])
                                    hardest_positive_ind.append(y[1])
                                    no_samples = False
                                    break
                        if not no_samples:
                            break
                    if no_samples:
                        print("There is no negative examples with distances greater than positive examples distances.")
                        exit(0)
            else:
                if self.num_hardest_samples is not None:
                    hard = np.argsort(anchor_positive_dist, axis=1)[:, -self.num_hardest_samples:]
                    ind = np.random.randint(self.num_hardest_samples, size=batch_size * num_samples)
                    hardest_positive_ind = hard[batch_size * num_samples * [True], ind]
                else:
                    hardest_positive_ind = np.argmax(anchor_positive_dist, axis=1)

            for i in range(batch_size):
                for j in range(num_samples):
                    c.append(s[i*num_samples+j])
                    rp.append(s[hardest_positive_ind[i*num_samples+j]])
                    rn.append(s[hardest_negative_ind[i*num_samples+j]])

        else:
            if self.semi_hard:
                for i in range(batch_size):
                # for i in range(1):
                    for j in range(num_samples):
                        for k in range(j+1, num_samples):
                            c.append(s[i*num_samples+j])
                            c.append(s[i*num_samples+k])
                            rp.append(s[i*num_samples+k])
                            rp.append(s[i*num_samples+j])
                            n, hrd = self.get_semi_hard_negative_ind(i, j, k, distances,
                                                                anchor_negative_dist,
                                                                batch_size, num_samples)
                            assert(n != i*num_samples+k)
                            rn.append(s[n])
                            hrds.append(hrd)
                            n, hrd = self.get_semi_hard_negative_ind(i, k, j, distances,
                                                                anchor_negative_dist,
                                                                batch_size, num_samples)
                            assert(n != i*num_samples+j)
                            rn.append(s[n])
                            hrds.append(hrd)
            else:
                for i in range(batch_size):
                    for j in range(num_samples):
                        for k in range(j + 1, num_samples):
                            c.append(s[i * num_samples + j])
                            c.append(s[i * num_samples + k])
                            rp.append(s[i * num_samples + k])
                            rp.append(s[i * num_samples + j])
                            rn.append(s[hardest_negative_ind[i * num_samples + j]])
                            rn.append(s[hardest_negative_ind[i * num_samples + k]])

        triplets = list(zip(c, rp, rn))
        np.random.shuffle(triplets)
        c = [el[0] for el in triplets]
        rp = [el[1] for el in triplets]
        rn = [el[2] for el in triplets]
        ratio = sum(hrds) / len(hrds)
        print("Ratio of semi-hard negative samples is %f" % ratio)
        return c, rp, rn

    def get_semi_hard_negative_ind(self, i, j, k, distances, anchor_negative_dist, batch_size, num_samples):
        anc_pos_dist = distances[i * num_samples + j, i * num_samples + k]
        neg_dists = anchor_negative_dist[i * num_samples + j]
        n_li_pre = sorted(list(zip(neg_dists, np.arange(batch_size * num_samples))), key=lambda el: el[0])
        n_li = list(filter(lambda x: x[1]<i*num_samples, n_li_pre)) + \
               list(filter(lambda x: x[1]>=(i+1)*num_samples, n_li_pre))
        for x in n_li:
            if x[0] > anc_pos_dist :
                return x[1], True
        return random.choice(n_li)[1], False

    @overrides
    def __call__(self, batch):
        if self.update_embeddings == 'on_validation':
            self.set_embeddings()
        if type(batch[0]) == list:
            context = [el[0] for el in batch]
            c = self.dict.make_toks(context, type="context")
            c = self.dict.make_ints(c)
            c_emb = self._net.predict_context_on_batch([c, c, c])
            if self.update_embeddings == 'online':
                self.update_contexts(c_emb, context)
            # response = [el[1] for el in batch]
            response = [list(el[1]) for el in batch]
            batch_size = len(response)
            ranking_length = len(response[0])
            # response = reduce(operator.concat, response)
            response = [x for el in response for x in el]
            if self.update_embeddings == 'online':
                r = self.dict.make_toks(response, type="response")
                r = self.dict.make_ints(r)
                r_embs = list(self._net.predict_response_on_batch([r, r, r]))
            elif self.update_embeddings == 'on_validation':
                r_embs = [self.dict.response2emb_vocab[el] for el in response]
            self.update_responses(r_embs, response)
            r_embs = [np.vstack(r_embs[i:batch_size*ranking_length:ranking_length])
                        for i in range(ranking_length)]
            y_pred = []
            for i in range(ranking_length):
                r_emb = r_embs[i]
                yp = np.sum(c_emb * r_emb, axis=1) / np.linalg.norm(c_emb, axis=1) / np.linalg.norm(r_emb, axis=1)
                y_pred.append(np.expand_dims(yp, axis=1))
            y_pred = np.hstack(y_pred)
            return y_pred

        elif type(batch[0]) == str:
            c_input = tokenize(batch)
            c_input = self.dict.make_ints(c_input)
            c_input_emb = self._net.predict_context_on_batch([c_input, c_input, c_input])

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

    def update_contexts(self, c_pred, labels):
        for i in range(len(labels)):
            self.dict.context2emb_vocab[labels[i]] = c_pred[i]

    def update_responses(self, r_pred, labels):
        for i in range(len(labels)):
            self.dict.response2emb_vocab[labels[i]] = r_pred[i]

    def set_embeddings(self):
        if self.dict.response2emb_vocab[0] is None:
            r = []
            for i in range(len(self.dict.response2toks_vocab)):
                r.append(self.dict.response2toks_vocab[i])
            r = self.dict.make_ints(r)
            response_embeddings = self._net.predict_response([r, r, r], 512)
            for i in range(len(self.dict.response2toks_vocab)):
                self.dict.response2emb_vocab[i] = response_embeddings[i]
        if self.dict.context2emb_vocab[0] is None:
            contexts = []
            for i in range(len(self.dict.context2toks_vocab)):
                contexts.append(self.dict.context2toks_vocab[i])
            contexts = self.dict.make_ints(contexts)
            context_embeddings = self._net.predict_context([contexts, contexts, contexts], 512)
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
