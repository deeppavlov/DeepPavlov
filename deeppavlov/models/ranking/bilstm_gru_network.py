from keras.layers import Input, GlobalMaxPooling1D, Lambda, Dense, GRU
from keras.models import Model
from keras import backend as K
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.models.ranking.bilstm_network import BiLSTMNetwork

log = get_logger(__name__)

@register('bilstm_gru_nn')
class BiLSTMGRUNetwork(BiLSTMNetwork):

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

    def __init__(self, *args, **kwargs):
        super(BiLSTMGRUNetwork, self).__init__(*args, **kwargs)

    def create_model(self):
        input = []
        if self.use_matrix:
            for i in range(self.num_context_turns + 1):
                input.append(Input(shape=(self.max_sequence_length,)))
            context = input[:self.num_context_turns]
            response = input[-1]
            emb_layer = self.embedding_layer()
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
        pooling_layer = GlobalMaxPooling1D(name="pooling")
        lstm_c = [pooling_layer(el) for el in lstm_c]
        lstm_r = pooling_layer(lstm_r)
        lstm_c = [Lambda(lambda x: K.expand_dims(x, 1))(el) for el in lstm_c]
        lstm_c = Lambda(lambda x: K.concatenate(x, 1))(lstm_c)
        gru_layer = GRU(2 * self.hidden_dim, name="gru")
        gru_c = gru_layer(lstm_c)

        if self.triplet_mode:
            dist = Lambda(self._pairwise_distances)([gru_c, lstm_r])
        else:
            dist = Lambda(self.diff_mult_dist)([gru_c, lstm_r])
            dist = Dense(1, activation='sigmoid', name="score_model")(dist)
        model = Model(context + [response], dist)
        return model

    def create_score_model(self):
        cr = self.model.inputs
        if self.triplet_mode:
            emb_c = self.model.get_layer("gru").output
            emb_r = self.model.get_layer("pooling").output
            dist_score = Lambda(lambda x: self.euclidian_dist(x), name="score_model")
            score = dist_score([emb_c, emb_r])
        else:
            score = self.model.get_layer("score_model").output
            score = Lambda(lambda x: 1. - K.squeeze(x, -1))(score)
        score = Lambda(lambda x: 1. - x)(score)
        model = Model(cr, score)
        return model
