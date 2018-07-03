from keras.layers import Input, LSTM, Embedding, GlobalMaxPooling1D, Lambda, subtract, Conv2D, Dense, Activation
from keras.layers.merge import Dot, Subtract, Add, Multiply
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.initializers import glorot_uniform, Orthogonal
from keras import backend as K
import tensorflow as tf
import numpy as np
from deeppavlov.core.models.tf_backend import TfModelMeta
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.layers import keras_layers
from pathlib import Path


log = get_logger(__name__)


class RankingNetwork(metaclass=TfModelMeta):

    def __init__(self, toks_num, emb_dict, use_matrix, max_sequence_length,
                 hidden_dim, learning_rate, margin, embedding_dim,
                 device_num=0, seed=None, type_of_weights="shared",
                 max_pooling=True, reccurent="bilstm", distance="cos_similarity",
                 max_token_length=None, char_emb_dim=None, embedding_level=None,
                 tok_dynamic_batch=False, char_dynamic_batch=False,
                 highway_on_top=False):
        self.distance = distance
        self.toks_num = toks_num
        self.emb_dict = emb_dict
        self.use_matrix = use_matrix
        self.seed = seed
        self.max_sequence_length = max_sequence_length
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.device_num = device_num
        self.type_of_weights = type_of_weights
        self.pooling = max_pooling
        self.recurrent = reccurent
        self.max_token_length = max_token_length
        self.embedding_level = embedding_level
        self.char_emb_dim = char_emb_dim
        self.tok_dynamic_batch = tok_dynamic_batch
        self.char_dynamic_batch = char_dynamic_batch
        self.highway_on_top = highway_on_top

        self.sess = self._config_session()
        K.set_session(self.sess)

        self.loss = self.triplet_loss
        self.optimizer = Adam(lr=self.learning_rate)
        self.obj_model = self.triplet_hinge_loss_model()
        self.obj_model.compile(loss=self.loss, optimizer=self.optimizer)
        self.score_model = Model(inputs=self.obj_model.input,
                                 outputs=self.obj_model.get_layer(name="score_model").get_output_at(0))
        self.context_embedding = Model(inputs=self.obj_model.input,
                                 outputs=self.obj_model.get_layer(name="pooling").get_output_at(0))
        self.response_embedding = Model(inputs=self.obj_model.input,
                                 outputs=self.obj_model.get_layer(name="pooling").get_output_at(1))

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
        if self.embedding_level is None or self.embedding_level == 'word':
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

    def triplet_hinge_loss_model(self):
        if self.embedding_level is None or self.embedding_level == 'word':
            if self.use_matrix:
                context = Input(shape=(self.max_sequence_length,))
                response_positive = Input(shape=(self.max_sequence_length,))
                response_negative = Input(shape=(self.max_sequence_length,))
                emb_layer_a, emb_layer_b = self.embedding_layer()
                emb_c = emb_layer_a(context)
                emb_rp = emb_layer_b(response_positive)
                emb_rn = emb_layer_b(response_negative)
            else:
                context = Input(shape=(self.max_sequence_length, self.embedding_dim,))
                response_positive = Input(shape=(self.max_sequence_length, self.embedding_dim,))
                response_negative = Input(shape=(self.max_sequence_length, self.embedding_dim,))
                emb_c = context
                emb_rp = response_positive
                emb_rn = response_negative
        elif self.embedding_level == 'char':
            if self.tok_dynamic_batch:
                msl = None
            else:
                msl = self.max_sequence_length
            if self.char_dynamic_batch:
                mtl = None
            else:
                mtl = self.tok_dynamic_batch

            context = Input(shape=(msl, mtl,))
            response_positive = Input(shape=(msl, mtl,))
            response_negative = Input(shape=(msl, mtl,))

            n_characters = self.max_sequence_length
            char_embedding_dim = self.char_emb_dim
            filter_widths = (3, 4, 5, 7)
            emb_layer = Embedding(n_characters,
                                     char_embedding_dim)

            emb_c = emb_layer(context)
            emb_rp = emb_layer(response_positive)
            emb_rn = emb_layer(response_negative)

            conv2d_layers = []
            for filter_width in filter_widths:
                conv2d_layers.append(Conv2D(char_embedding_dim,
                                      (1, filter_width),
                                      padding='same'))
            conv_results_list = []
            for cl in conv2d_layers:
                conv_results_list.append(cl(emb_c))
                # units = Concatenate(conv_results_list, axis=3)
            emb_c = Lambda(lambda x: K.concatenate(x, axis=3))(conv_results_list)
            emb_c = Lambda(lambda x: K.max(x, axis=2))(emb_c)

            conv_results_list = []
            for cl in conv2d_layers:
                conv_results_list.append(cl(emb_rp))
                # units = Concatenate(conv_results_list, axis=3)
            emb_rp = Lambda(lambda x: K.concatenate(x, axis=3))(conv_results_list)
            emb_rp = Lambda(lambda x: K.max(x, axis=2))(emb_rp)

            conv_results_list = []
            for cl in conv2d_layers:
                conv_results_list.append(cl(emb_rn))
                # units = Concatenate(conv_results_list, axis=3)
            emb_rn = Lambda(lambda x: K.concatenate(x, axis=3))(conv_results_list)
            emb_rn = Lambda(lambda x: K.max(x, axis=2))(emb_rn)

            if self.highway_on_top:
                dense1 = Dense(char_embedding_dim * len(filter_widths))
                dense2 = Dense(char_embedding_dim * len(filter_widths))

                sigmoid_gate = dense1(emb_c)
                sigmoid_gate = Activation('sigmoid')(sigmoid_gate)
                deeper_units = dense2(emb_c)
                emb_c = Add()([Multiply()([sigmoid_gate, deeper_units]),
                             Multiply()([Lambda(lambda x: K.constant(1.) - x)(sigmoid_gate), emb_c])])
                emb_c = Activation('relu')(emb_c)

                sigmoid_gate = dense1(emb_rp)
                sigmoid_gate = Activation('sigmoid')(sigmoid_gate)
                deeper_units = dense2(emb_rp)
                emb_rp = Add()([Multiply()([sigmoid_gate, deeper_units]),
                              Multiply()([Lambda(lambda x: K.constant(1.) - x)(sigmoid_gate), emb_rp])])
                emb_rp = Activation('relu')(emb_rp)

                sigmoid_gate = dense1(emb_rn)
                sigmoid_gate = Activation('sigmoid')(sigmoid_gate)
                deeper_units = dense2(emb_rn)
                emb_rn = Add()([Multiply()([sigmoid_gate, deeper_units]),
                              Multiply()([Lambda(lambda x: K.constant(1.) - x)(sigmoid_gate), emb_rn])])
                emb_rn = Activation('relu')(emb_rn)




            # emb_layer = Lambda(lambda x:
            #                    keras_layers.character_embedding_network(x, n_characters=self.max_token_length,
            #                                                             char_embedding_dim= self.char_emb_dim))

            # char_emb_layer = keras_layers.CharEmbeddingCNN(n_characters=self.max_token_length,
            #                                            char_embedding_dim=self.char_emb_dim)

            # char_emb_layer = Lambda(lambda x: keras_layers.char_embedding_cnn(x,
            #                                                                   n_characters=self.max_token_length,
            #                                                                   char_embedding_dim=self.char_emb_dim))

            # emb_c = char_emb_layer(context)
            # emb_rp = char_emb_layer(response_positive)
            # emb_rn = char_emb_layer(response_negative)

        lstm_layer_a, lstm_layer_b = self.lstm_layer()
        lstm_c = lstm_layer_a(emb_c)
        lstm_rp = lstm_layer_b(emb_rp)
        lstm_rn = lstm_layer_b(emb_rn)
        if self.pooling:
            pooling_layer = GlobalMaxPooling1D(name="pooling")
            lstm_c = pooling_layer(lstm_c)
            lstm_rp = pooling_layer(lstm_rp)
            lstm_rn = pooling_layer(lstm_rn)
        if self.distance == "euclidian":
            # cosine_layer = Dot(normalize=False, axes=-1, name="score_model")
            # dist_pos = Subtract([lstm_c, lstm_rp])
            # dist_neg = Subtract([lstm_c, lstm_rn])
            # cosine_pos = cosine_layer([dist_pos, dist_pos])
            # cosine_neg = cosine_layer([dist_neg, dist_neg])
            # score_diff = Subtract()([cosine_neg, cosine_pos])
            dist_score = Lambda(self.euclidian_dist,
                                output_shape=self.euclidian_dist_output_shape,
                                name="score_model")
            dist_pos = dist_score([lstm_c, lstm_rp])
            dist_neg = dist_score([lstm_c, lstm_rn])
            score_diff = Subtract()([dist_neg, dist_pos])
        elif self.distance == "cos_similarity":
            cosine_layer = Dot(normalize=True, axes=-1, name="score_model")
            dist_pos = cosine_layer([lstm_c, lstm_rp])
            dist_neg = cosine_layer([lstm_c, lstm_rn])
            score_diff = Subtract()([dist_pos, dist_neg])
        model = Model([context, response_positive, response_negative], score_diff)
        return model

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

    def triplet_loss(self, y_true, y_pred):
        """Triplet loss function"""

        return K.mean(K.maximum(self.margin - y_pred, 0.), axis=-1)

    def train_on_batch(self, batch):
        if self.embedding_level is None or self.embedding_level == 'word':
            if self.use_matrix:
                self.obj_model.train_on_batch(x=[np.asarray(x) for x in batch[0]], y=np.asarray(batch[1]))
            else:
                a, b, c = batch[0]
                a = self.emb_dict.get_embs(a)
                b = self.emb_dict.get_embs(b)
                c = self.emb_dict.get_embs(c)
                self.obj_model.train_on_batch(x=[a, b, c], y=np.asarray(batch[1]))
        elif self.embedding_level == 'char':
            self.obj_model.train_on_batch(x=[np.asarray(x) for x in batch[0]], y=np.asarray(batch[1]))

    def predict_on_batch(self, batch):
        if self.embedding_level is None or self.embedding_level == 'word':
            if self.use_matrix:
                return self.score_model.predict_on_batch(x=batch)
            else:
                a, b, c = batch
                a = self.emb_dict.get_embs(a)
                b = self.emb_dict.get_embs(b)
                c = self.emb_dict.get_embs(c)
                return self.score_model.predict_on_batch(x=[a, b, c])
        elif self.embedding_level == 'char':
            return self.score_model.predict_on_batch(x=batch)

    def predict_context_on_batch(self, batch):
        if self.embedding_level is None or self.embedding_level == 'word':
            if self.use_matrix:
                return self.context_embedding.predict_on_batch(x=batch)
            else:
                a, b, c = batch
                a = self.emb_dict.get_embs(a)
                b = self.emb_dict.get_embs(b)
                c = self.emb_dict.get_embs(c)
                return self.context_embedding.predict_on_batch(x=[a, b, c])
        elif self.embedding_level == 'char':
            return self.context_embedding.predict_on_batch(x=batch)

    def predict_context(self, batch, bs):
        if self.embedding_level is None or self.embedding_level == 'word':
            if self.use_matrix:
                return self.context_embedding.predict(x=batch, batch_size=bs)
            else:
                cont_embs = []
                num_batches = len(batch[0]) // bs
                for i in range(num_batches):
                    a = batch[0][i * bs:(i + 1) * bs]
                    b = batch[1][i * bs:(i + 1) * bs]
                    c = batch[2][i * bs:(i + 1) * bs]
                    cont_embs.append(self.predict_context_on_batch([a, b, c]))
                if len(batch[0]) % bs != 0:
                    a = batch[0][num_batches * bs:]
                    b = batch[1][num_batches * bs:]
                    c = batch[2][num_batches * bs:]
                    cont_embs.append(self.predict_context_on_batch([a, b, c]))
                cont_embs = np.vstack(cont_embs)
            return cont_embs
        elif self.embedding_level == 'char':
            return self.context_embedding.predict(x=batch, batch_size=bs)

    def predict_response_on_batch(self, batch):
        if self.embedding_level is None or self.embedding_level == 'word':
            if self.use_matrix:
                return self.response_embedding.predict_on_batch(x=batch)
            else:
                a, b, c = batch
                a = self.emb_dict.get_embs(a)
                b = self.emb_dict.get_embs(b)
                c = self.emb_dict.get_embs(c)
                return self.response_embedding.predict_on_batch(x=[a, b, c])
        elif self.embedding_level == 'char':
            return self.response_embedding.predict_on_batch(x=batch)

    def predict_response(self, batch, bs):
        if self.embedding_level is None or self.embedding_level == 'word':
            if self.use_matrix:
                return self.response_embedding.predict(x=batch, batch_size=bs)
            else:
                resp_embs = []
                num_batches = len(batch[0]) // bs
                for i in range(num_batches):
                    a = batch[0][i * bs:(i + 1) * bs]
                    b = batch[1][i * bs:(i + 1) * bs]
                    c = batch[2][i * bs:(i + 1) * bs]
                    resp_embs.append(self.predict_response_on_batch([a, b, c]))
                if len(batch[0]) % bs != 0:
                    a = batch[0][num_batches * bs:]
                    b = batch[1][num_batches * bs:]
                    c = batch[2][num_batches * bs:]
                    resp_embs.append(self.predict_response_on_batch([a, b, c]))
                resp_embs = np.vstack(resp_embs)
                return resp_embs
        elif self.embedding_level == 'char':
            return self.response_embedding.predict(x=batch, batch_size=bs)

