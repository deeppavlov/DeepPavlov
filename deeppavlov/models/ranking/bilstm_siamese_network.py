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

from logging import getLogger
from typing import List

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import losses
from tensorflow.keras.initializers import glorot_uniform, Orthogonal
from tensorflow.keras.layers import (Input, LSTM, Embedding, GlobalMaxPooling1D, Lambda, Dense, Layer, Multiply,
                                     Bidirectional)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework.ops import Tensor

from deeppavlov.core.common.registry import register
from deeppavlov.models.ranking.keras_siamese_model import KerasSiameseModel

log = getLogger(__name__)


@register('bilstm_nn')
class BiLSTMSiameseNetwork(KerasSiameseModel):
    """The class implementing a siamese neural network with BiLSTM and max pooling.

    There is a possibility to use a binary cross-entropy loss as well as
    a triplet loss with random or hard negative sampling.

    Args:
        len_vocab: A size of the vocabulary to build embedding layer.
        seed: Random seed.
        shared_weights: Whether to use shared weights in the model to encode ``contexts`` and ``responses``.
        embedding_dim: Dimensionality of token (word) embeddings.
        reccurent: A type of the RNN cell. Possible values are ``lstm`` and ``bilstm``.
        hidden_dim: Dimensionality of the hidden state of the RNN cell. If ``reccurent`` equals ``bilstm``
            ``hidden_dim`` should be doubled to get the actual dimensionality.
        max_pooling: Whether to use max-pooling operation to get ``context`` (``response``) vector representation.
            If ``False``, the last hidden state of the RNN will be used.
        triplet_loss: Whether to use a model with triplet loss.
            If ``False``, a model with crossentropy loss will be used.
        margin: A margin parameter for triplet loss. Only required if ``triplet_loss`` is set to ``True``.
        hard_triplets: Whether to use hard triplets sampling to train the model
            i.e. to choose negative samples close to positive ones.
            If set to ``False`` random sampling will be used.
            Only required if ``triplet_loss`` is set to ``True``.
    """

    def __init__(self,
                 len_vocab: int,
                 seed: int = None,
                 shared_weights: bool = True,
                 embedding_dim: int = 300,
                 reccurent: str = "bilstm",
                 hidden_dim: int = 300,
                 max_pooling: bool = True,
                 triplet_loss: bool = True,
                 margin: float = 0.1,
                 hard_triplets: bool = False,
                 *args,
                 **kwargs) -> None:

        self.toks_num = len_vocab
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.shared_weights = shared_weights
        self.pooling = max_pooling
        self.recurrent = reccurent
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.hard_triplets = hard_triplets
        self.triplet_mode = triplet_loss

        super(BiLSTMSiameseNetwork, self).__init__(*args, **kwargs)

    def compile(self) -> None:
        optimizer = Adam(lr=self.learning_rate)
        if self.triplet_mode:
            loss = self._triplet_loss
        else:
            loss = losses.binary_crossentropy
        self.model.compile(loss=loss, optimizer=optimizer)
        self.score_model = self.create_score_model()

    def load_initial_emb_matrix(self) -> None:
        log.info("[initializing new `{}`]".format(self.__class__.__name__))
        if self.use_matrix:
            if self.shared_weights:
                self.model.get_layer(name="embedding").set_weights([self.emb_matrix])
            else:
                self.model.get_layer(name="embedding_a").set_weights([self.emb_matrix])
                self.model.get_layer(name="embedding_b").set_weights([self.emb_matrix])

    def embedding_layer(self) -> Layer:
        out = Embedding(self.toks_num,
                        self.embedding_dim,
                        input_length=self.max_sequence_length,
                        trainable=True, name="embedding")
        return out

    def lstm_layer(self) -> Layer:
        if self.pooling:
            ret_seq = True
        else:
            ret_seq = False
        ker_in = glorot_uniform(seed=self.seed)
        rec_in = Orthogonal(seed=self.seed)
        if self.recurrent == "bilstm" or self.recurrent is None:
            out = Bidirectional(LSTM(self.hidden_dim,
                                     input_shape=(self.max_sequence_length, self.embedding_dim,),
                                     kernel_initializer=ker_in,
                                     recurrent_initializer=rec_in,
                                     return_sequences=ret_seq), merge_mode='concat')
        elif self.recurrent == "lstm":
            out = LSTM(self.hidden_dim,
                       input_shape=(self.max_sequence_length, self.embedding_dim,),
                       kernel_initializer=ker_in,
                       recurrent_initializer=rec_in,
                       return_sequences=ret_seq)
        return out

    def create_model(self) -> Model:
        if self.use_matrix:
            context = Input(shape=(self.max_sequence_length,))
            response = Input(shape=(self.max_sequence_length,))
            if self.shared_weights:
                emb_layer_a = self.embedding_layer()
                emb_layer_b = emb_layer_a
            else:
                emb_layer_a = self.embedding_layer()
                emb_layer_b = self.embedding_layer()
            emb_c = emb_layer_a(context)
            emb_r = emb_layer_b(response)
        else:
            context = Input(shape=(self.max_sequence_length, self.embedding_dim,))
            response = Input(shape=(self.max_sequence_length, self.embedding_dim,))
            emb_c = context
            emb_r = response

        if self.shared_weights:
            lstm_layer_a = self.lstm_layer()
            lstm_layer_b = lstm_layer_a
        else:
            lstm_layer_a = self.lstm_layer()
            lstm_layer_b = self.lstm_layer()
        lstm_c = lstm_layer_a(emb_c)
        lstm_r = lstm_layer_b(emb_r)
        if self.pooling:
            pooling_layer = GlobalMaxPooling1D(name="sentence_embedding")
            lstm_c = pooling_layer(lstm_c)
            lstm_r = pooling_layer(lstm_r)

        if self.triplet_mode:
            dist = Lambda(self._pairwise_distances)([lstm_c, lstm_r])
        else:
            dist = Lambda(self._diff_mult_dist)([lstm_c, lstm_r])
            dist = Dense(1, activation='sigmoid', name="score_model")(dist)
        model = Model([context, response], dist)
        return model

    def create_score_model(self) -> Model:
        cr = self.model.inputs
        if self.triplet_mode:
            emb_c = self.model.get_layer("sentence_embedding").get_output_at(0)
            emb_r = self.model.get_layer("sentence_embedding").get_output_at(1)
            dist_score = Lambda(lambda x: self._euclidian_dist(x), name="score_model")
            score = dist_score([emb_c, emb_r])
        else:
            score = self.model.get_layer("score_model").output
            score = Lambda(lambda x: 1. - K.squeeze(x, -1))(score)
        score = Lambda(lambda x: 1. - x)(score)
        model = Model(cr, score)
        return model

    def _diff_mult_dist(self, inputs: List[Tensor]) -> Tensor:
        input1, input2 = inputs
        a = K.abs(input1 - input2)
        b = Multiply()(inputs)
        return K.concatenate([input1, input2, a, b])

    def _euclidian_dist(self, x_pair: List[Tensor]) -> Tensor:
        x1_norm = K.l2_normalize(x_pair[0], axis=1)
        x2_norm = K.l2_normalize(x_pair[1], axis=1)
        diff = x1_norm - x2_norm
        square = K.square(diff)
        _sum = K.sum(square, axis=1)
        _sum = K.clip(_sum, min_value=1e-12, max_value=None)
        dist = K.sqrt(_sum) / 2.
        return dist

    def _pairwise_distances(self, inputs: List[Tensor]) -> Tensor:
        emb_c, emb_r = inputs
        bs = K.shape(emb_c)[0]
        embeddings = K.concatenate([emb_c, emb_r], 0)
        dot_product = K.dot(embeddings, K.transpose(embeddings))
        square_norm = K.batch_dot(embeddings, embeddings, axes=1)
        distances = K.transpose(square_norm) - 2.0 * dot_product + square_norm
        distances = distances[0:bs, bs:bs+bs]
        distances = K.clip(distances, 0.0, None)
        mask = K.cast(K.equal(distances, 0.0), K.dtype(distances))
        distances = distances + mask * 1e-16
        distances = K.sqrt(distances)
        distances = distances * (1.0 - mask)
        return distances

    def _triplet_loss(self, labels: Tensor, pairwise_dist: Tensor) -> Tensor:
        y_true = K.squeeze(labels, axis=1)
        """Triplet loss function"""
        if self.hard_triplets:
            triplet_loss = self._batch_hard_triplet_loss(y_true, pairwise_dist)
        else:
            triplet_loss = self._batch_all_triplet_loss(y_true, pairwise_dist)
        return triplet_loss

    def _batch_all_triplet_loss(self, y_true: Tensor, pairwise_dist: Tensor) -> Tensor:
        anchor_positive_dist = K.expand_dims(pairwise_dist, 2)
        anchor_negative_dist = K.expand_dims(pairwise_dist, 1)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.margin
        mask = self._get_triplet_mask(y_true, pairwise_dist)
        triplet_loss = mask * triplet_loss
        triplet_loss = K.clip(triplet_loss, 0.0, None)
        valid_triplets = K.cast(K.greater(triplet_loss, 1e-16), K.dtype(triplet_loss))
        num_positive_triplets = K.sum(valid_triplets)
        triplet_loss = K.sum(triplet_loss) / (num_positive_triplets + 1e-16)
        return triplet_loss

    def _batch_hard_triplet_loss(self, y_true: Tensor, pairwise_dist: Tensor) -> Tensor:
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(y_true, pairwise_dist)
        anchor_positive_dist = mask_anchor_positive * pairwise_dist
        hardest_positive_dist = K.max(anchor_positive_dist, axis=1, keepdims=True)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(y_true, pairwise_dist)
        anchor_negative_dist = mask_anchor_negative * pairwise_dist
        mask_anchor_negative = self._get_semihard_anchor_negative_triplet_mask(anchor_negative_dist,
                                                                               hardest_positive_dist,
                                                                               mask_anchor_negative)
        max_anchor_negative_dist = K.max(pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)
        hardest_negative_dist = K.min(anchor_negative_dist, axis=1, keepdims=True)
        triplet_loss = K.clip(hardest_positive_dist - hardest_negative_dist + self.margin, 0.0, None)
        triplet_loss = K.mean(triplet_loss)
        return triplet_loss

    def _get_triplet_mask(self, y_true: Tensor, pairwise_dist: Tensor) -> Tensor:
        # mask label(a) != label(p)
        mask1 = K.expand_dims(K.equal(K.expand_dims(y_true, 0), K.expand_dims(y_true, 1)), 2)
        mask1 = K.cast(mask1, K.dtype(pairwise_dist))
        # mask a == p
        mask2 = K.expand_dims(K.not_equal(pairwise_dist, 0.0), 2)
        mask2 = K.cast(mask2, K.dtype(pairwise_dist))
        # mask label(n) == label(a)
        mask3 = K.expand_dims(K.not_equal(K.expand_dims(y_true, 0), K.expand_dims(y_true, 1)), 1)
        mask3 = K.cast(mask3, K.dtype(pairwise_dist))
        return mask1 * mask2 * mask3

    def _get_anchor_positive_triplet_mask(self, y_true: Tensor, pairwise_dist: Tensor) -> Tensor:
        # mask label(a) != label(p)
        mask1 = K.equal(K.expand_dims(y_true, 0), K.expand_dims(y_true, 1))
        mask1 = K.cast(mask1, K.dtype(pairwise_dist))
        # mask a == p
        mask2 = K.not_equal(pairwise_dist, 0.0)
        mask2 = K.cast(mask2, K.dtype(pairwise_dist))
        return mask1 * mask2

    def _get_anchor_negative_triplet_mask(self, y_true: Tensor, pairwise_dist: Tensor) -> Tensor:
        # mask label(n) == label(a)
        mask = K.not_equal(K.expand_dims(y_true, 0), K.expand_dims(y_true, 1))
        mask = K.cast(mask, K.dtype(pairwise_dist))
        return mask

    def _get_semihard_anchor_negative_triplet_mask(self, negative_dist: Tensor,
                                                   hardest_positive_dist: Tensor,
                                                   mask_negative: Tensor) -> Tensor:
        # mask max(dist(a,p)) < dist(a,n)
        mask = K.greater(negative_dist, hardest_positive_dist)
        mask = K.cast(mask, K.dtype(negative_dist))
        mask_semihard = K.cast(K.expand_dims(K.greater(K.sum(mask, 1), 0.0), 1), K.dtype(negative_dist))
        mask = mask_negative * (1 - mask_semihard) + mask * mask_semihard
        return mask

    def _predict_on_batch(self, batch: List[np.ndarray]) -> np.ndarray:
        return self.score_model.predict_on_batch(x=batch)
