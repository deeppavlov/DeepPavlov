from keras.layers import Input, LSTM, Embedding, GlobalMaxPooling1D, Lambda, subtract, Conv2D, Dense, Activation, GRU
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
from deeppavlov.core.common.registry import register
from deeppavlov.models.ranking.siamese_embeddings_network import SiameseEmbeddingsNetwork

log = get_logger(__name__)

@register('bilstm_gru_nn')
class BiLSTMGRUNetwork(SiameseEmbeddingsNetwork):

    """Class to perform context-response matching with neural networks.

    Args:
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
                 len_vocab: int,
                 max_sequence_length: int,
                 num_context_turns: int,
                 len_char_vocab: int = None,
                 max_token_length: int = None,
                 seed: int = None,
                 token_embeddings: bool = True,
                 use_matrix: bool = False,
                 tok_dynamic_batch: bool = False,
                 embedding_dim: int = 300,
                 char_embeddings: bool = False,
                 char_dynamic_batch: bool = False,
                 char_emb_dim: int = 32,
                 hidden_dim: int = 300,
                 **kwargs):

        self.toks_num = len_vocab
        self.num_context_turns = num_context_turns
        self.use_matrix = use_matrix
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.token_embeddings = token_embeddings
        self.char_embeddings = char_embeddings
        self.chars_num = len_char_vocab
        self.char_emb_dim = char_emb_dim
        if tok_dynamic_batch:
            self.max_sequence_length = None
        else:
            self.max_sequence_length = max_sequence_length
        if char_dynamic_batch:
            self.max_token_length = None
        else:
            self.max_token_length = max_token_length




    def embedding_layer(self):
        out = Embedding(self.toks_num,
                        self.embedding_dim,
                        input_length=self.max_sequence_length,
                        trainable=True, name="embedding")
        return out

    def lstm_layer(self):
        """Create a LSTM layer of a model."""
        ker_in = glorot_uniform(seed=self.seed)
        rec_in = Orthogonal(seed=self.seed)
        out = Bidirectional(LSTM(self.hidden_dim,
                            input_shape=(self.max_sequence_length, self.embedding_dim,),
                            kernel_initializer=ker_in,
                            recurrent_initializer=rec_in,
                            return_sequences=True), merge_mode='concat')
        return out

    def embeddings_model(self):
        input = []
        if self.use_matrix:
            for i in range(self.num_context_turns + 1):
                input.append(Input(shape=(self.max_sequence_length,)))
            context = input[:self.num_context_turns]
            response = input[-1]
            emb_layer = self.embedding_layer()
            # for i in range(self.num_context_turns):
            #     emb_c = emb_layer()
            emb_c = [emb_layer(el) for el in context]
            emb_r = emb_layer(response)
        else:
            for i in range(self.num_context_turns + 1):
                input.append(Input(shape=(self.max_sequence_length, self.embedding_dim,)))
            context = input[:self.num_context_turns]
            response = input[-1]
            emb_c = context
            emb_r = response
        lstm_layer = self.lstm_layer()
        lstm_c = [lstm_layer(el) for el in emb_c]
        lstm_r = lstm_layer(emb_r)
        pooling_layer = GlobalMaxPooling1D()
        lstm_c = [pooling_layer(el) for el in lstm_c]
        lstm_r = pooling_layer(lstm_r)
        lstm_c = [Lambda(lambda x: K.expand_dims(x, 1))(el) for el in lstm_c]
        lstm_c = Lambda(lambda x: K.concatenate(x, 1))(lstm_c)
        gru_layer = GRU(2 * self.hidden_dim)
        gru_c = gru_layer(lstm_c)
        model = Model(input, [gru_c, lstm_r])
        return model