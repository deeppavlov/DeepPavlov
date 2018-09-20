from keras.layers import Input, LSTM, Embedding, GlobalMaxPooling1D, Lambda, subtract, Conv2D, Dense, Activation
from keras.layers.merge import Dot, Subtract, Add, Multiply
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.initializers import glorot_uniform, Orthogonal
from keras import losses
from keras import backend as K
import tensorflow as tf
import numpy as np
from deeppavlov.core.models.tf_backend import TfModelMeta
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.layers import keras_layers
from pathlib import Path
from deeppavlov.core.commands.utils import expand_path
from typing import List, Callable
from deeppavlov.core.models.keras_model import KerasModel
from deeppavlov.core.common.registry import register
from deeppavlov.models.ranking.siamese_network import SiameseNetwork

log = get_logger(__name__)

@register('bilstm_nn')
class BiLSTMNetwork(KerasModel, metaclass=TfModelMeta):

    """Class to perform context-response matching with neural networks.

    Args:
        toks_num: A size of `tok2int` vocabulary to build embedding layer.
        chars_num: A size of `char2int` vocabulary to build character-level embedding layer.

        learning_rate: Learning rate.
        device_num: A number of a device to perform model training on if several devices are available in a system.
        seed: Random seed.
        shared_weights: Whether to use shared weights in the model to encode contexts and responses.
        triplet_loss: Whether to use a model with triplet loss.
            If ``False``, a model with crossentropy loss will be used.
        margin: A margin parameter for triplet loss. Only required if ``triplet_mode`` is set to ``True``.
        token_embeddings: Whether to use token (word) embeddings in the model.
        max_sequence_length: A maximum length of a sequence in tokens.
            Longer sequences will be truncated and shorter ones will be padded.
        tok_dynamic_batch:  Whether to use dynamic batching. If ``True``, a maximum length of a sequence for a batch
            will be equal to the maximum of all sequences lengths from this batch,
            but not higher than ``max_sequence_length``.
        embedding_dim: Dimensionality of token (word) embeddings.
        char_embeddings: Whether to use character-level token (word) embeddings in the model.
        max_token_length: A maximum length of a token for representing it by a character-level embedding.
        char_dynamic_batch: Whether to use dynamic batching for character-level embeddings.
            If ``True``, a maximum length of a token for a batch
            will be equal to the maximum of all tokens lengths from this batch,
            but not higher than ``max_token_length``.
        char_emb_dim: Dimensionality of character-level embeddings.
        reccurent: A type of the RNN cell. Possible values are ``lstm`` and ``bilstm``.
        hidden_dim: Dimensionality of the hidden state of the RNN cell. If ``reccurent`` equals ``bilstm``
            to get the actual dimensionality ``hidden_dim`` should be doubled.
        max_pooling: Whether to use max-pooling operation to get context (response) vector representation.
            If ``False``, the last hidden state of the RNN will be used.
        hard_triplets: Whether to use hard triplets sampling to train the model
            i.e. to choose negative samples close to positive ones.

                    toks_num: A size of `tok2int` vocabulary to build embedding layer.
        chars_num: A size of `char2int` vocabulary to build character-level embedding layer.

        learning_rate: Learning rate.
        device_num: A number of a device to perform model training on if several devices are available in a system.
        seed: Random seed.
        shared_weights: Whether to use shared weights in the model to encode contexts and responses.
        triplet_mode: Whether to use a model with triplet loss.
            If ``False``, a model with crossentropy loss will be used.
        margin: A margin parameter for triplet loss. Only required if ``triplet_mode`` is set to ``True``.
        token_embeddings: Whether to use token (word) embeddings in the model.
        max_sequence_length: A maximum length of a sequence in tokens.
            Longer sequences will be truncated and shorter ones will be padded.
        tok_dynamic_batch:  Whether to use dynamic batching. If ``True``, a maximum length of a sequence for a batch
            will be equal to the maximum of all sequences lengths from this batch,
            but not higher than ``max_sequence_length``.
        embedding_dim: Dimensionality of token (word) embeddings.
        char_embeddings: Whether to use character-level token (word) embeddings in the model.
        max_token_length: A maximum length of a token for representing it by a character-level embedding.
        char_dynamic_batch: Whether to use dynamic batching for character-level embeddings.
            If ``True``, a maximum length of a token for a batch
            will be equal to the maximum of all tokens lengths from this batch,
            but not higher than ``max_token_length``.
        char_emb_dim: Dimensionality of character-level embeddings.
        reccurent: A type of the RNN cell. Possible values are ``lstm`` and ``bilstm``.
        hidden_dim: Dimensionality of the hidden state of the RNN cell. If ``reccurent`` equals ``bilstm``
            to get the actual dimensionality ``hidden_dim`` should be doubled.
        max_pooling: Whether to use max-pooling operation to get context (response) vector representation.
            If ``False``, the last hidden state of the RNN will be used.
        hard_triplets_sampling: Whether to use hard triplets sampling to train the model
            i.e. to choose negative samples close to positive ones.
        hardest_positives: Whether to use only one hardest positive sample per each anchor sample.
            It is only used when ``hard_triplets_sampling`` is set to ``True``.
        semi_hard_negatives: Whether hard negative samples should be further away from anchor samples
            than positive samples or not. It is only used when ``hard_triplets_sampling`` is set to ``True``.
        num_hardest_negatives: It is only used when ``hard_triplets_sampling`` is set to ``True``
            and ``semi_hard_negatives`` is set to ``False``.
    """

    def __init__(self,
                 use_matrix: bool,
                 len_vocab: int,
                 max_sequence_length: int,
                 len_char_vocab: int = None,
                 max_token_length: int = None,
                 seed: int = None,
                 shared_weights: bool = True,
                 token_embeddings: bool = True,
                 tok_dynamic_batch: bool = False,
                 embedding_dim: int = 300,
                 char_embeddings: bool = False,
                 char_dynamic_batch: bool = False,
                 char_emb_dim: int = 32,
                 reccurent: str = "bilstm",
                 hidden_dim: int = 300,
                 max_pooling: bool = True,
                 emb_matrix: np.ndarray = None,
                 learning_rate: float = 1e-3,
                 device_num: int = 0,
                 triplet_loss: bool = True,
                 margin: float = 0.1,
                 highway_on_top: bool = False,
                 hard_triplets: bool = False,
                 **kwargs):

        self.toks_num = len_vocab
        self.use_matrix = use_matrix
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.shared_weights = shared_weights
        self.pooling = max_pooling
        self.recurrent = reccurent
        self.token_embeddings = token_embeddings
        self.char_embeddings = char_embeddings
        self.chars_num = len_char_vocab
        self.char_emb_dim = char_emb_dim
        self.emb_matrix = emb_matrix
        self.learning_rate = learning_rate
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.device_num = device_num
        self.highway_on_top = highway_on_top
        self.hard_triplets = hard_triplets
        self.triplet_mode = triplet_loss

        if tok_dynamic_batch:
            self.max_sequence_length = None
        else:
            self.max_sequence_length = max_sequence_length
        if char_dynamic_batch:
            self.max_token_length = None
        else:
            self.max_token_length = max_token_length

        self.optimizer = Adam(lr=self.learning_rate)
        if self.triplet_mode:
            self.loss = self.triplet_loss
        else:
            self.loss = losses.binary_crossentropy
        self.obj_model = self.bilstm_model()
        self.obj_model.compile(loss=self.loss, optimizer=self.optimizer)
        self.score_model = self.prediction_model()
        # self.context_embedding = Model(inputs=self.embeddings.inputs,
        #                          outputs=self.embeddings.outputs[0])
        # self.response_embedding = Model(inputs=self.embeddings.inputs,
        #                          outputs=self.embeddings.outputs[1])

    def load(self, load_path):
        log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
        self.obj_model.load_weights(str(load_path))

    def save(self, save_path):
        log.info("[saving `{}`]".format(self.__class__.__name__))
        self.obj_model.save_weights(str(save_path))

    def load_initial_emb_matrix(self):
        log.info("[initializing new `{}`]".format(self.__class__.__name__))
        if self.use_matrix:
            if self.token_embeddings and not self.char_embeddings:
                if self.shared_weights:
                    self.obj_model.get_layer(name="embedding").set_weights([self.emb_matrix])
                else:
                    self.obj_model.get_layer(name="embedding_a").set_weights([self.emb_matrix])
                    self.obj_model.get_layer(name="embedding_b").set_weights([self.emb_matrix])

    def embedding_layer(self):
        out = Embedding(self.toks_num,
                        self.embedding_dim,
                        input_length=self.max_sequence_length,
                        trainable=True, name="embedding")
        return out

    def lstm_layer(self):
        """Create a LSTM layer of a model."""
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

    def bilstm_model(self):
        if self.token_embeddings and not self.char_embeddings:
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
        elif not self.token_embeddings and self.char_embeddings:
            context = Input(shape=(self.max_sequence_length, self.max_token_length,))
            response = Input(shape=(self.max_sequence_length, self.max_token_length,))

            char_cnn_layer = keras_layers.char_emb_cnn_func(n_characters=self.chars_num,
                                                            char_embedding_dim=self.char_emb_dim)
            emb_c = char_cnn_layer(context)
            emb_r = char_cnn_layer(response)

        elif self.token_embeddings and self.char_embeddings:
            context = Input(shape=(self.max_sequence_length, self.max_token_length,))
            response = Input(shape=(self.max_sequence_length, self.max_token_length,))

            if self.use_matrix:
                c_tok = Lambda(lambda x: x[:,:,0])(context)
                r_tok = Lambda(lambda x: x[:,:,0])(response)
                emb_layer_a, emb_layer_b = self.embedding_layer()
                emb_c = emb_layer_a(c_tok)
                emb_rp = emb_layer_b(r_tok)
                c_char = Lambda(lambda x: x[:,:,1:])(context)
                r_char = Lambda(lambda x: x[:,:,1:])(response)
            else:
                c_tok = Lambda(lambda x: x[:,:,:self.embedding_dim])(context)
                r_tok = Lambda(lambda x: x[:,:,:self.embedding_dim])(response)
                emb_c = c_tok
                emb_rp = r_tok
                c_char = Lambda(lambda x: x[:,:,self.embedding_dim:])(context)
                r_char = Lambda(lambda x: x[:,:,self.embedding_dim:])(response)

            char_cnn_layer = keras_layers.char_emb_cnn_func(n_characters=self.chars_num,
                                                            char_embedding_dim=self.char_emb_dim)

            emb_c_char = char_cnn_layer(c_char)
            emb_r_char = char_cnn_layer(r_char)

            emb_c = Lambda(lambda x: K.concatenate(x, axis=-1))([emb_c, emb_c_char])
            emb_r = Lambda(lambda x: K.concatenate(x, axis=-1))([emb_rp, emb_r_char])

        if self.shared_weights:
            lstm_layer_a = self.lstm_layer()
            lstm_layer_b = lstm_layer_a
        else:
            lstm_layer_a = self.lstm_layer()
            lstm_layer_b = self.lstm_layer()
        lstm_c = lstm_layer_a(emb_c)
        lstm_r = lstm_layer_b(emb_r)
        if self.pooling:
            pooling_layer = GlobalMaxPooling1D(name="pooling")
            lstm_c = pooling_layer(lstm_c)
            lstm_r = pooling_layer(lstm_r)

        if self.triplet_mode:
            dist = Lambda(self._pairwise_distances)([lstm_c, lstm_r])
        else:
            dist = Lambda(self.diff_mult_dist)([lstm_c, lstm_r])
            dist = Dense(1, activation='sigmoid', name="score_model")(dist)
        model = Model([context, response], dist)
        return model

    def prediction_model(self):
        cr = self.obj_model.inputs
        if self.triplet_mode:
            emb_c, emb_r = self.obj_model.get_layer("pooling").outputs
            dist_score = Lambda(lambda x: self.euclidian_dist(x), name="score_model")
            score = dist_score([emb_c, emb_r])
        else:
            score = self.obj_model.get_layer("score_model").output
            score = Lambda(lambda x: 1. - K.squeeze(x, -1))(score)
        score = Lambda(lambda x: 1. - x)(score)
        model = Model(cr, score)
        return model

    def diff_mult_dist(self, inputs):
        input1, input2 = inputs
        a = K.abs(input1-input2)
        b = Multiply()(inputs)
        return K.concatenate([input1, input2, a, b])

    def euclidian_dist(self, x_pair):
        x1_norm = K.l2_normalize(x_pair[0], axis=1)
        x2_norm = K.l2_normalize(x_pair[1], axis=1)
        diff = x1_norm - x2_norm
        square = K.square(diff)
        sum = K.sum(square, axis=1)
        sum = K.clip(sum, min_value=1e-12, max_value=None)
        dist = K.sqrt(sum)
        return dist

    def _pairwise_distances(self, inputs):
        emb_c, emb_r = inputs
        bs = K.shape(emb_c)[0]
        embeddings = K.concatenate([emb_c, emb_r], 0)
        dot_product = K.dot(embeddings, K.transpose(embeddings))
        square_norm = K.batch_dot(embeddings, embeddings, axes=1)
        distances = K.transpose(square_norm) - 2.0 * dot_product + square_norm
        distances = K.slice(distances, (0, bs), (bs, bs))
        distances = K.clip(distances, 0.0, None)
        mask = K.cast(K.equal(distances, 0.0), K.dtype(distances))
        distances = distances + mask * 1e-16
        distances = K.sqrt(distances)
        distances = distances * (1.0 - mask)
        return distances

    def triplet_loss(self, labels, pairwise_dist):
        y_true = K.squeeze(labels, axis=1)
        """Triplet loss function"""
        if self.hard_triplets:
            triplet_loss = self.batch_hard_triplet_loss(y_true, pairwise_dist)
        else:
            triplet_loss = self.batch_all_triplet_loss(y_true, pairwise_dist)
        return triplet_loss

    def batch_all_triplet_loss(self, y_true, pairwise_dist):
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

    def batch_hard_triplet_loss(self, y_true, pairwise_dist):
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

    def _get_triplet_mask(self, y_true, pairwise_dist):
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

    def _get_anchor_positive_triplet_mask(self, y_true, pairwise_dist):
        # mask label(a) != label(p)
        mask1 = K.equal(K.expand_dims(y_true, 0), K.expand_dims(y_true, 1))
        mask1 = K.cast(mask1, K.dtype(pairwise_dist))
        # mask a == p
        mask2 = K.not_equal(pairwise_dist, 0.0)
        mask2 = K.cast(mask2, K.dtype(pairwise_dist))
        return mask1 * mask2

    def _get_anchor_negative_triplet_mask(self, y_true, pairwise_dist):
        # mask label(n) == label(a)
        mask = K.not_equal(K.expand_dims(y_true, 0), K.expand_dims(y_true, 1))
        mask = K.cast(mask, K.dtype(pairwise_dist))
        return mask

    def _get_semihard_anchor_negative_triplet_mask(self, negative_dist, hardest_positive_dist, mask_negative):
        # mask max(dist(a,p)) < dist(a,n)
        mask = K.greater(negative_dist, hardest_positive_dist)
        mask = K.cast(mask, K.dtype(negative_dist))
        mask_semihard = K.cast(K.expand_dims(K.greater(K.sum(mask, 1), 0.0), 1), K.dtype(negative_dist))
        mask = mask_negative * (1 - mask_semihard) + mask * mask_semihard
        return mask

    def train_on_batch(self, batch, y):
        # b = [x for el in batch for x in el]
        if self.token_embeddings and not self.char_embeddings:
            # self.obj_model.train_on_batch(x=[np.asarray(x) for x in batch], y=np.asarray(y))
            loss = self.obj_model.train_on_batch(x=list(batch), y=np.asarray(y))
        elif not self.token_embeddings and self.char_embeddings:
            loss = self.obj_model.train_on_batch(x=[np.asarray(x) for x in batch], y=np.asarray(y))
        elif self.token_embeddings and self.char_embeddings:
            if self.use_matrix:
                loss = self.obj_model.train_on_batch(x=[np.asarray(x) for x in batch], y=np.asarray(y))
            else:
                b = [x[0] for x in batch]
                loss = self.obj_model.train_on_batch(x=b, y=np.asarray(y))
        return loss

    def __call__(self, batch):
        if self.token_embeddings and not self.char_embeddings:
            return self.score_model.predict_on_batch(x=batch)
        elif not self.token_embeddings and self.char_embeddings:
            return self.score_model.predict_on_batch(x=batch)
        elif self.token_embeddings and self.char_embeddings:
            if self.use_matrix:
                return self.score_model.predict_on_batch(x=batch)
            else:
                b = [batch[i][:,:,0] for i in range(len(batch))]
                b = [np.concatenate([b[i], batch[i][:,:,1:]], axis=2) for i in range(len(batch))]
                return self.score_model.predict_on_batch(x=b)

    def predict_embedding_on_batch(self, batch, type='context'):
        if type == 'context':
            embedding = self.context_embedding
        elif type == 'response':
            embedding = self.response_embedding
        if self.token_embeddings and not self.char_embeddings:
            return embedding.predict_on_batch(x=batch)
        elif not self.token_embeddings and self.char_embeddings:
            return embedding.predict_on_batch(x=batch)
        elif self.token_embeddings and self.char_embeddings:
            if self.use_matrix:
                return embedding.predict_on_batch(x=batch)
            else:
                b = [batch[i][:,:,0] for i in range(len(batch))]
                b = [np.concatenate([b[i], batch[i][:,:,1:]], axis=2) for i in range(len(batch))]
                return embedding.predict_on_batch(x=b)

    def reset(self):
        pass