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
from deeppavlov.models.ranking.emb_dict import EmbDict
from deeppavlov.core.common.log import get_logger
from typing import Union, List, Tuple, Dict

log = get_logger(__name__)


@register('ranking_model')
class RankingModel(NNModel):
    """Class to perform ranking.

    Args:
        vocab_name: A key word that indicates which subclass
            of the :class:`deeppavlov.models.ranking.ranking_dict.RankingDict` to use.
        hard_triplets_sampling: Whether to use hard triplets sampling to train the model
            i.e. to choose negative samples close to positive ones.
        hardest_positives: Whether to use only one hardest positive sample per each anchor sample.
        semi_hard_negatives: Whether hard negative samples should be further away from anchor samples
            than positive samples or not.
        update_embeddings: Whether to store and update context and response embeddings or not.
        interact_pred_num: The number of the most relevant contexts and responses
            which model returns in the `interact` regime.
        **kwargs: Other parameters.
    """

    def __init__(self,
                 vocab_name,
                 hard_triplets_sampling: bool = False,
                 hardest_positives: bool = False,
                 semi_hard_negatives: bool = False,
                 num_hardest_negatives: int = None,
                 update_embeddings: bool = False,
                 interact_pred_num: int = 3,
                 **kwargs):

        # Parameters for parent classes
        save_path = kwargs.get('save_path', None)
        load_path = kwargs.get('load_path', None)
        train_now = kwargs.get('train_now', None)
        mode = kwargs.get('mode', None)

        super().__init__(save_path=save_path, load_path=load_path,
                         train_now=train_now, mode=mode)

        self.hard_triplets_sampling = hard_triplets_sampling
        self.hardest_positives = hardest_positives
        self.semi_hard_negatives = semi_hard_negatives
        self.num_hardest_negatives = num_hardest_negatives
        self.upd_embs = update_embeddings
        self.interact_pred_num = interact_pred_num
        self.train_now = train_now
        self.vocab_name = vocab_name

        opt = deepcopy(kwargs)

        if self.vocab_name == "insurance":
            dict_parameter_names = list(inspect.signature(InsuranceDict.__init__).parameters)
            dict_parameters = {par: opt[par] for par in dict_parameter_names if par in opt}
            self.dict = InsuranceDict(**dict_parameters, update_embeddings=update_embeddings)

        embdict_parameter_names = list(inspect.signature(EmbDict.__init__).parameters)
        embdict_parameters = {par: opt[par] for par in embdict_parameter_names if par in opt}
        self.embdict= EmbDict(**embdict_parameters)

        network_parameter_names = list(inspect.signature(RankingNetwork.__init__).parameters)
        self.network_parameters = {par: opt[par] for par in network_parameter_names if par in opt}

        self.load()

        train_parameters_names = list(inspect.signature(self._net.train_on_batch).parameters)
        self.train_parameters = {par: opt[par] for par in train_parameters_names if par in opt}

    @overrides
    def load(self):
        """Load the model from the last checkpoint."""
        if not self.load_path.exists():
            log.info("[initializing new `{}`]".format(self.__class__.__name__))
            self.dict.init_from_scratch()
            self.embdict.init_from_scratch(self.dict.tok2int_vocab)
            if hasattr(self.dict, 'char2int_vocab'):
                chars_num = len(self.dict.char2int_vocab)
            else:
                chars_num = 0
            self._net = RankingNetwork(chars_num=chars_num,
                                       toks_num=len(self.dict.tok2int_vocab),
                                       emb_dict=self.embdict,
                                       **self.network_parameters)
            self._net.init_from_scratch(self.embdict.emb_matrix)
        else:
            log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
            self.dict.load()
            self.embdict.load()
            if hasattr(self.dict, 'char2int_vocab'):
                chars_num = len(self.dict.char2int_vocab)
            else:
                chars_num = 0
            self._net = RankingNetwork(chars_num=chars_num,
                                       toks_num=len(self.dict.tok2int_vocab),
                                       emb_dict=self.embdict,
                                       **self.network_parameters)
            self._net.load(self.load_path)

    @overrides
    def save(self):
        """Save the model."""
        log.info('[saving model to {}]'.format(self.save_path.resolve()))
        self._net.save(self.save_path)
        if self.upd_embs:
            self.set_embeddings()
        self.dict.save()
        self.embdict.save()

    @check_attr_true('train_now')
    def train_on_batch(self, x: List[List[Tuple[int, int]]], y: List[int]):
        """Train the model on a batch."""
        if self.upd_embs:
            self.reset_embeddings()
        if self.hard_triplets_sampling:
            b = self.make_hard_triplets(x, y, self._net)
            y = np.ones(len(b[0][0]))
        else:
            b = self.make_batch(x)
            for i in range(len(x[0])):
                c = self.dict.make_toks(b[i][0], type="context")
                c = self.dict.make_ints(c)
                b[i][0] = c
                r = self.dict.make_toks(b[i][1], type="response")
                r = self.dict.make_ints(r)
                b[i][1] = r
        self._net.train_on_batch(b, y)

    def  make_batch(self, x):
        sample_len = len(x[0])
        b = []
        for i in range(sample_len):
            c = []
            r = []
            for el in x:
                c.append(el[i][0])
                r.append(el[i][1])
            b.append([c, r])
        return b

    def make_hard_triplets(self, x, y, net):
        samples = [[s[1] for s in el] for el in x]
        labels = y
        batch_size = len(samples)
        num_samples = len(samples[0])
        samp = [y for el in samples for y in el]
        s = self.dict.make_toks(samp, type="context")
        s = self.dict.make_ints(s)

        embeddings = net.predict_embedding([s, s], 512, type='context')
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
        if self.num_hardest_negatives is not None:
            hard = np.argsort(anchor_negative_dist, axis=1)[:, :self.num_hardest_negatives]
            ind = np.random.randint(self.num_hardest_negatives, size=batch_size * num_samples)
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

            if self.semi_hard_negatives:
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
                        log.error("There are no negative examples with distances"
                                  " greater than positive examples distances.")
                        exit(0)
            else:
                if self.num_hardest_negatives is not None:
                    hard = np.argsort(anchor_positive_dist, axis=1)[:, -self.num_hardest_negatives:]
                    ind = np.random.randint(self.num_hardest_negatives, size=batch_size * num_samples)
                    hardest_positive_ind = hard[batch_size * num_samples * [True], ind]
                else:
                    hardest_positive_ind = np.argmax(anchor_positive_dist, axis=1)

            for i in range(batch_size):
                for j in range(num_samples):
                    c.append(s[i*num_samples+j])
                    rp.append(s[hardest_positive_ind[i*num_samples+j]])
                    rn.append(s[hardest_negative_ind[i*num_samples+j]])

        else:
            if self.semi_hard_negatives:
                for i in range(batch_size):
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
        log.info("Ratio of semi-hard negative samples is %f" % ratio)
        return [(c, rp), (c, rn)]

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

    def __call__(self, batch: Union[List[List[Tuple[int, int]]], List[str]]) ->\
            Union[np.ndarray, Dict[str, List[str]]]:
        """Make a prediction on a batch."""
        if type(batch[0]) == list:
            y_pred = []
            b = self.make_batch(batch)
            for el in b:
                c = self.dict.make_toks(el[0], type="context")
                c = self.dict.make_ints(c)
                r = self.dict.make_toks(el[1], type="response")
                r = self.dict.make_ints(r)
                yp = self._net.predict_score_on_batch([c, r])
                y_pred.append(yp)
            y_pred = np.hstack(y_pred)
            return y_pred

        elif type(batch[0]) == str:
            c_input = tokenize(batch)
            c_input = self.dict.make_ints(c_input)
            c_input_emb = self._net.predict_embedding_on_batch([c_input, c_input], type='context')

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

    def update_embeddings(self, batch):
        sample_len = len(batch[0])
        labels_cont = []
        labels_resp = []
        cont = []
        resp = []
        for i in range(sample_len):
            lc = []
            lr = []
            for el in batch:
                lc.append(el[i][0])
                lr.append(el[i][1])
            labels_cont.append(lc)
            labels_resp.append(lr)
        for i in range(sample_len):
            c = self.dict.make_toks(labels_cont[i], type="context")
            c = self.dict.make_ints(c)
            cont.append(c)
            r = self.dict.make_toks(labels_resp[i], type="response")
            r = self.dict.make_ints(r)
            resp.append(r)
        for el in zip(labels_cont, cont):
            c_emb = self._net.predict_embedding_on_batch([el[1], el[1]], type='context')
            for i in range(len(el[0])):
                self.dict.context2emb_vocab[el[0][i]] = c_emb[i]
        for el in zip(labels_resp, resp):
            r_emb = self._net.predict_embedding_on_batch([el[1], el[1]], type='response')
            for i in range(len(el[0])):
                self.dict.response2emb_vocab[el[0][i]] = r_emb[i]

    def set_embeddings(self):
        if self.dict.response2emb_vocab[0] is None:
            r = []
            for i in range(len(self.dict.response2toks_vocab)):
                r.append(self.dict.response2toks_vocab[i])
            r = self.dict.make_ints(r)
            response_embeddings = self._net.predict_embedding([r, r], 512, type='response')
            for i in range(len(self.dict.response2toks_vocab)):
                self.dict.response2emb_vocab[i] = response_embeddings[i]
        if self.dict.context2emb_vocab[0] is None:
            c = []
            for i in range(len(self.dict.context2toks_vocab)):
                c.append(self.dict.context2toks_vocab[i])
            c = self.dict.make_ints(c)
            context_embeddings = self._net.predict_embedding([c, c], 512, type='context')
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
