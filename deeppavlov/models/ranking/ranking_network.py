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


log = get_logger(__name__)


class RankingNetwork(metaclass=TfModelMeta):

    def __init__(self, chars_num, toks_num, emb_dict, use_matrix, max_sequence_length,
                 hidden_dim, learning_rate, margin, embedding_dim,
                 device_num=0, seed=None, type_of_weights="shared",
                 max_pooling=True, reccurent="bilstm", distance="cos_similarity",
                 max_token_length=None, char_emb_dim=None, embedding_level=None,
                 tok_dynamic_batch=False, char_dynamic_batch=False,
                 highway_on_top=False, type_of_model=None):
        self.distance = distance

        self.toks_num = toks_num
        self.emb_dict = emb_dict
        self.use_matrix = use_matrix
        self.seed = seed
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.device_num = device_num
        self.type_of_weights = type_of_weights
        self.pooling = max_pooling
        self.recurrent = reccurent
        self.embedding_level = embedding_level
        self.chars_num = chars_num
        self.char_emb_dim = char_emb_dim
        self.highway_on_top = highway_on_top
        self.type_of_model = type_of_model
        if tok_dynamic_batch:
            self.max_sequence_length = None
        else:
            self.max_sequence_length = max_sequence_length
        if char_dynamic_batch:
            self.max_token_length = None
        else:
            self.max_token_length = max_token_length

        self.sess = self._config_session()
        K.set_session(self.sess)

        self.optimizer = Adam(lr=self.learning_rate)
        self.duplet = self.duplet_model()
        if self.type_of_model is None or self.type_of_model == 'triplet':
            self.loss = self.triplet_loss
            self.obj_model = self.triplet_model()
        elif self.type_of_model == 'duplet':
            self.loss = losses.binary_crossentropy
            self.obj_model = self.duplet
        self.obj_model.compile(loss=self.loss, optimizer=self.optimizer)
        self.score_model = self.duplet
        self.context_embedding = self.duplet.get_layer(name="pooling").get_output_at(0)
        self.response_embedding = self.duplet.get_layer(name="pooling").get_output_at(1)

        self.context_embedding = Model(inputs=self.duplet.inputs,
                                 outputs=self.duplet.get_layer(name="pooling").get_output_at(0))
        self.response_embedding = Model(inputs=self.duplet.inputs,
                                 outputs=self.duplet.get_layer(name="pooling").get_output_at(1))

    def _config_session(self):
        """
        Configure session for particular device
        Returns:
            tensorflow.Session
        """
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(self.device_num)
        return tf.Session(config=config)

    def load(self, path):
        log.info("[initializing `{}` from saved]".format(self.__class__.__name__))
        self.obj_model.load_weights(path)

    def save(self, path):
        log.info("[saving `{}`]".format(self.__class__.__name__))
        self.obj_model.save_weights(path)
        self.context_embedding.save(str(Path(path).parent / 'sen_emb_model.h5'))

    def init_from_scratch(self, emb_matrix):
        log.info("[initializing new `{}`]".format(self.__class__.__name__))
        if self.embedding_level is None or self.embedding_level == 'token':
            if self.use_matrix:
                if self.type_of_weights == "shared":
                    self.obj_model.get_layer(name="embedding").set_weights([emb_matrix])
                if self.type_of_weights == "separate":
                    self.obj_model.get_layer(name="embedding_a").set_weights([emb_matrix])
                    self.obj_model.get_layer(name="embedding_b").set_weights([emb_matrix])

    def embedding_layer(self):
        if self.type_of_weights == "shared":
            out_a = Embedding(self.toks_num,
                            self.embedding_dim,
                            input_length=self.max_sequence_length,
                            trainable=True, name="embedding")
            return out_a, out_a
        elif self.type_of_weights == "separate":
            out_a = Embedding(self.toks_num,
                            self.embedding_dim,
                            input_length=self.max_sequence_length,
                            trainable=True, name="embedding_a")
            out_b = Embedding(self.toks_num,
                            self.embedding_dim,
                            input_length=self.max_sequence_length,
                            trainable=True, name="embedding_b")
            return out_a, out_b

    def lstm_layer(self):
        """Create a LSTM layer of a model."""
        if self.pooling:
            ret_seq = True
        else:
            ret_seq = False
        ker_in = glorot_uniform(seed=self.seed)
        rec_in = Orthogonal(seed=self.seed)
        if self.type_of_weights == "shared":
            if self.recurrent == "bilstm" or self.recurrent is None:
                out_a = Bidirectional(LSTM(self.hidden_dim,
                                    input_shape=(self.max_sequence_length, self.embedding_dim,),
                                    kernel_initializer=ker_in,
                                    recurrent_initializer=rec_in,
                                    return_sequences=ret_seq), merge_mode='concat')
            elif self.recurrent == "lstm":
                out_a = LSTM(self.hidden_dim,
                           input_shape=(self.max_sequence_length, self.embedding_dim,),
                           kernel_initializer=ker_in,
                           recurrent_initializer=rec_in,
                           return_sequences=ret_seq)
            return out_a, out_a
        elif self.type_of_weights == "separate":
            if self.recurrent == "bilstm" or self.recurrent is None:
                out_a = Bidirectional(LSTM(self.hidden_dim,
                                    input_shape=(self.max_sequence_length, self.embedding_dim,),
                                    kernel_initializer=ker_in,
                                    recurrent_initializer=rec_in,
                                    return_sequences=ret_seq), merge_mode='concat')
                out_b = Bidirectional(LSTM(self.hidden_dim,
                                    input_shape=(self.max_sequence_length, self.embedding_dim,),
                                    kernel_initializer=ker_in,
                                    recurrent_initializer=rec_in,
                                    return_sequences=ret_seq), merge_mode='concat')
            elif self.recurrent == "lstm":
                out_a = LSTM(self.hidden_dim,
                           input_shape=(self.max_sequence_length, self.embedding_dim,),
                           kernel_initializer=ker_in,
                           recurrent_initializer=rec_in,
                           return_sequences=ret_seq)
                out_b = LSTM(self.hidden_dim,
                           input_shape=(self.max_sequence_length, self.embedding_dim,),
                           kernel_initializer=ker_in,
                           recurrent_initializer=rec_in,
                           return_sequences=ret_seq)
            return out_a, out_b

    def triplet_loss(self, y_true, y_pred):
        """Triplet loss function"""
        return K.mean(K.maximum(self.margin - y_pred, 0.), axis=-1)

    def duplet_model(self):
        if self.embedding_level is None or self.embedding_level == 'token':
            if self.use_matrix:
                context = Input(shape=(self.max_sequence_length,))
                response = Input(shape=(self.max_sequence_length,))
                emb_layer_a, emb_layer_b = self.embedding_layer()
                emb_c = emb_layer_a(context)
                emb_r = emb_layer_b(response)
            else:
                context = Input(shape=(self.max_sequence_length, self.embedding_dim,))
                response = Input(shape=(self.max_sequence_length, self.embedding_dim,))
                emb_c = context
                emb_r = response
        elif self.embedding_level == 'char':
            context = Input(shape=(self.max_sequence_length, self.max_token_length,))
            response = Input(shape=(self.max_sequence_length, self.max_token_length,))

            char_cnn_layer = keras_layers.char_emb_cnn_func(n_characters=self.chars_num,
                                                            char_embedding_dim=self.char_emb_dim)
            emb_c = char_cnn_layer(context)
            emb_r = char_cnn_layer(response)

        elif self.embedding_level == 'token_and_char':
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

        lstm_layer_a, lstm_layer_b = self.lstm_layer()
        lstm_c = lstm_layer_a(emb_c)
        lstm_r = lstm_layer_b(emb_r)
        if self.pooling:
            pooling_layer = GlobalMaxPooling1D(name="pooling")
            lstm_c = pooling_layer(lstm_c)
            lstm_r = pooling_layer(lstm_r)

        if self.distance == "cos_similarity":
            cosine_layer = Dot(normalize=True, axes=-1, name="score_model")
            score = cosine_layer([lstm_c, lstm_r])
            score = Lambda(lambda x: -x)(score)
        elif self.distance == "euclidian":
            dist_score = Lambda(self.euclidian_dist,
                                output_shape=self.euclidian_dist_output_shape,
                                name="score_model")
            score = dist_score([lstm_c, lstm_r])
        elif self.distance == "sigmoid":
            dist = Lambda(self.diff_mult_dist)([lstm_c, lstm_r])
            score = Dense(1, activation='sigmoid', name="score_model")(dist)

        model = Model([context, response], score)
        return model

    def triplet_model(self):
        duplet = self.duplet
        c_shape = K.int_shape(duplet.inputs[0])
        r_shape = K.int_shape(duplet.inputs[1])
        c1 = Input(batch_shape=c_shape)
        r1 = Input(batch_shape=r_shape)
        c2 = Input(batch_shape=c_shape)
        r2 = Input(batch_shape=r_shape)
        score1 = duplet([c1, r1])
        score2 = duplet([c2, r2])
        score_diff = Subtract()([score2, score1])
        model = Model([c1, r1, c2, r2], score_diff)
        return model

    def diff_mult_dist(self, inputs):
        input1, input2 = inputs
        a = K.abs(input1-input2)
        b = Multiply()(inputs)
        return K.concatenate([input1, input2, a, b])

    def euclidian_dist_output_shape(self, input_shape):
        return (input_shape[0][0], 1)

    def euclidian_dist(self, x_pair):
        x1_norm = K.l2_normalize(x_pair[0], axis=1)
        x2_norm = K.l2_normalize(x_pair[1], axis=1)
        diff = subtract([x1_norm, x2_norm])
        square = K.square(diff)
        sum = K.sum(square, axis=1)
        sum = K.clip(sum, min_value=1e-12, max_value=None)
        dist = K.sqrt(sum)
        return dist

    def train_on_batch(self, batch, y):
        batch = [x for el in batch for x in el]
        if self.embedding_level is None or self.embedding_level == 'token':
            if self.use_matrix:
                self.obj_model.train_on_batch(x=[np.asarray(x) for x in batch], y=np.asarray(y))
            else:
                b = batch
                for i in range(len(b)):
                    b[i] = self.emb_dict.get_embs(b[i])
                self.obj_model.train_on_batch(x=b, y=np.asarray(y))
        elif self.embedding_level == 'char':
            self.obj_model.train_on_batch(x=[np.asarray(x) for x in batch], y=np.asarray(y))
        elif self.embedding_level == 'token_and_char':
            if self.use_matrix:
                self.obj_model.train_on_batch(x=[np.asarray(x) for x in batch], y=np.asarray(y))
            else:
                b = [x[0] for x in batch]
                for i in range(len(b)):
                    b[i] = self.emb_dict.get_embs(b[i])
                self.obj_model.train_on_batch(x=b, y=np.asarray(y))

    def predict_score_on_batch(self, batch):
        if self.embedding_level is None or self.embedding_level == 'token':
            if self.use_matrix:
                return self.score_model.predict_on_batch(x=batch)
            else:
                b = batch
                for i in range(len(b)):
                    b[i] = self.emb_dict.get_embs(b[i])
                return self.score_model.predict_on_batch(x=b)
        elif self.embedding_level == 'char':
            return self.score_model.predict_on_batch(x=batch)
        elif self.embedding_level == 'token_and_char':
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
        if self.embedding_level is None or self.embedding_level == 'token':
            if self.use_matrix:
                return embedding.predict_on_batch(x=batch)
            else:
                b = batch
                b = [self.emb_dict.get_embs(el) for el in b]
                return embedding.predict_on_batch(x=b)
        elif self.embedding_level == 'char':
            return embedding.predict_on_batch(x=batch)
        elif self.embedding_level == 'token_and_char':
            if self.use_matrix:
                return embedding.predict_on_batch(x=batch)
            else:
                b = [self.emb_dict.get_embs(batch[i][:,:,0]) for i in range(len(batch))]
                b = [np.concatenate([b[i], batch[i][:,:,1:]], axis=2) for i in range(len(batch))]
                return embedding.predict_on_batch(x=b)

    def predict_embedding(self, batch, bs, type='context'):
                num_batches = len(batch[0]) // bs
                embs = []
                for i in range(num_batches):
                    b = [batch[j][i * bs:(i + 1) * bs] for j in range(len(batch))]
                    embs.append(self.predict_embedding_on_batch(b))
                if len(batch[0]) % bs != 0:
                    b = [batch[j][num_batches * bs:] for j in range(len(batch))]
                    embs.append(self.predict_embedding_on_batch(b, type=type))
                embs = np.vstack(embs)
                return embs

    # def triplet_hinge_loss_model(self):
    #     if self.embedding_level is None or self.embedding_level == 'token':
    #         if self.use_matrix:
    #             context = Input(shape=(self.max_sequence_length,))
    #             response_positive = Input(shape=(self.max_sequence_length,))
    #             response_negative = Input(shape=(self.max_sequence_length,))
    #             emb_layer_a, emb_layer_b = self.embedding_layer()
    #             emb_c = emb_layer_a(context)
    #             emb_rp = emb_layer_b(response_positive)
    #             emb_rn = emb_layer_b(response_negative)
    #         else:
    #             context = Input(shape=(self.max_sequence_length, self.embedding_dim,))
    #             response_positive = Input(shape=(self.max_sequence_length, self.embedding_dim,))
    #             response_negative = Input(shape=(self.max_sequence_length, self.embedding_dim,))
    #             emb_c = context
    #             emb_rp = response_positive
    #             emb_rn = response_negative
    #     elif self.embedding_level == 'char':
    #         context = Input(shape=(self.max_sequence_length, self.max_token_length,))
    #         response_positive = Input(shape=(self.max_sequence_length, self.max_token_length,))
    #         response_negative = Input(shape=(self.max_sequence_length, self.max_token_length,))
    #
    #         char_cnn_layer = keras_layers.char_emb_cnn_func(n_characters=self.chars_num,
    #                                                         char_embedding_dim=self.char_emb_dim)
    #         emb_c = char_cnn_layer(context)
    #         emb_rp = char_cnn_layer(response_positive)
    #         emb_rn = char_cnn_layer(response_negative)
    #
    #     elif self.embedding_level == 'token_and_char':
    #         context = Input(shape=(self.max_sequence_length, self.max_token_length,))
    #         response_positive = Input(shape=(self.max_sequence_length, self.max_token_length,))
    #         response_negative = Input(shape=(self.max_sequence_length, self.max_token_length,))
    #
    #         if self.use_matrix:
    #             c_tok = Lambda(lambda x: x[:,:,0])(context)
    #             rp_tok = Lambda(lambda x: x[:,:,0])(response_positive)
    #             rn_tok = Lambda(lambda x: x[:,:,0])(response_negative)
    #             emb_layer_a, emb_layer_b = self.embedding_layer()
    #             emb_c = emb_layer_a(c_tok)
    #             emb_rp = emb_layer_b(rp_tok)
    #             emb_rn = emb_layer_b(rn_tok)
    #             c_char = Lambda(lambda x: x[:,:,1:])(context)
    #             rp_char = Lambda(lambda x: x[:,:,1:])(response_positive)
    #             rn_char = Lambda(lambda x: x[:,:,1:])(response_negative)
    #         else:
    #             c_tok = Lambda(lambda x: x[:,:,:self.embedding_dim])(context)
    #             rp_tok = Lambda(lambda x: x[:,:,:self.embedding_dim])(response_positive)
    #             rn_tok = Lambda(lambda x: x[:,:,:self.embedding_dim])(response_negative)
    #             emb_c = c_tok
    #             emb_rp = rp_tok
    #             emb_rn = rn_tok
    #             c_char = Lambda(lambda x: x[:,:,self.embedding_dim:])(context)
    #             rp_char = Lambda(lambda x: x[:,:,self.embedding_dim:])(response_positive)
    #             rn_char = Lambda(lambda x: x[:,:,self.embedding_dim:])(response_negative)
    #
    #         char_cnn_layer = keras_layers.char_emb_cnn_func(n_characters=self.chars_num,
    #                                                         char_embedding_dim=self.char_emb_dim)
    #
    #         emb_c_char = char_cnn_layer(c_char)
    #         emb_rp_char = char_cnn_layer(rp_char)
    #         emb_rn_char = char_cnn_layer(rn_char)
    #
    #         emb_c = Lambda(lambda x: K.concatenate(x, axis=-1))([emb_c, emb_c_char])
    #         emb_rp = Lambda(lambda x: K.concatenate(x, axis=-1))([emb_rp, emb_rp_char])
    #         emb_rn = Lambda(lambda x: K.concatenate(x, axis=-1))([emb_rn, emb_rn_char])
    #
    #     lstm_layer_a, lstm_layer_b = self.lstm_layer()
    #     lstm_c = lstm_layer_a(emb_c)
    #     lstm_rp = lstm_layer_b(emb_rp)
    #     lstm_rn = lstm_layer_b(emb_rn)
    #     if self.pooling:
    #         pooling_layer = GlobalMaxPooling1D(name="pooling")
    #         lstm_c = pooling_layer(lstm_c)
    #         lstm_rp = pooling_layer(lstm_rp)
    #         lstm_rn = pooling_layer(lstm_rn)
    #     if self.distance == "euclidian":
    #         dist_score = Lambda(self.euclidian_dist,
    #                             output_shape=self.euclidian_dist_output_shape,
    #                             name="score_model")
    #         dist_pos = dist_score([lstm_c, lstm_rp])
    #         dist_neg = dist_score([lstm_c, lstm_rn])
    #         score_diff = Subtract()([dist_neg, dist_pos])
    #     elif self.distance == "cos_similarity":
    #         cosine_layer = Dot(normalize=True, axes=-1, name="score_model")
    #         dist_pos = cosine_layer([lstm_c, lstm_rp])
    #         dist_neg = cosine_layer([lstm_c, lstm_rn])
    #         score_diff = Subtract()([dist_pos, dist_neg])
    #     model = Model([context, response_positive, response_negative], score_diff)
    #     return model
