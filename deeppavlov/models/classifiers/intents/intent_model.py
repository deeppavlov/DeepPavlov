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

import numpy as np

from keras.layers import Dense, Input, concatenate, Activation, Concatenate, Reshape
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM, GRU
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
from keras.models import Model
from keras.regularizers import l2

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.keras_model import KerasModel
from deeppavlov.models.classifiers.intents.utils import labels2onehot, proba2labels
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from deeppavlov.models.classifiers.intents.utils import md5_hashsum
from deeppavlov.models.tokenizers.nltk_tokenizer import NLTKTokenizer
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.layers.keras_layers import additive_self_attention, multiplicative_self_attention


log = get_logger(__name__)


@register('intent_model')
class KerasIntentModel(KerasModel):
    """
    Class implements keras model for intent recognition task for multi-class multi-label data
    """
    def __init__(self, **kwargs):
        """
        Initialize and train vocabularies, initializes embedder, tokenizer,
        and then initialize model using parameters from opt dictionary (from config),
        if model is being initialized from saved

        Args:
            vocabs: dictionary of considered vocabularies
            opt: model parameters for network and learning
            model_path: path to model serialization dir or file.
                            It is always an empty string and is ignored if it is not set in json config.
            model_dir: name of a serialization dir, can be default or set in json config
            model_file: name of a serialization file (usually binary model file),
                            can be default or set in json config
            embedder: instance of FasttextEmbedder class
            tokenizer: instance of NLTKTokenizer class
            **kwargs:
        """
        super().__init__(**kwargs) # self.opt initialized in here

        self.tokenizer = self.opt.get('tokenizer')
        self.fasttext_model = self.opt.get('embedder')
        self.opt.pop("vocabs")
        self.opt.pop("embedder")
        self.opt.pop("tokenizer")

        if self.opt.get('classes'):
            self.classes = list(np.sort(np.array(list(self.opt.get('classes')))))
            self.opt['classes'] = self.classes
        else:
            # self.classes = list(np.sort(np.array(list(self.opt.get('vocabs')["classes_vocab"].keys()))))
            self.classes = list(self.opt.get('vocabs')["classes_vocab"].keys())
            self.opt['classes'] = self.classes
        self.n_classes = len(self.classes)
        if self.n_classes == 0:
            ConfigError("Please, provide vocabulary with considered intents.")

        self.opt['embedding_size'] = self.fasttext_model.dim

        if self.fasttext_model.load_path:
            current_fasttext_md5 = md5_hashsum([self.fasttext_model.load_path])

        # Parameters required to init model
        params = {"model_name": self.opt.get('model_name'),
                  "optimizer_name": self.opt.get('optimizer'),
                  "loss_name": self.opt.get('loss'),
                  "lear_rate": self.opt.get('lear_rate'),
                  "lear_rate_decay": self.opt.get('lear_rate_decay')}

        self.model = self.load(**params)
        self._init_params()

        # Check if md5 hash sum of current loaded fasttext model
        # is equal to saved
        try:
            self.opt['fasttext_md5']
        except KeyError:
            self.opt['fasttext_md5'] = current_fasttext_md5
        else:
            if self.opt['fasttext_md5'] != current_fasttext_md5:
                raise ConfigError(
                    "Given fasttext model does NOT match fasttext model used previously to train loaded model")

    def _init_params(self):

        # list of changeable params
        changeable_params = {"confident_threshold": 0.5,
                             "optimizer": "Adam",
                             "lear_rate": 1e-2,
                             "lear_rate_decay": 0.,
                             "loss": "binary_crossentropy",
                             "coef_reg_cnn": 0.,
                             "coef_reg_den": 0.,
                             "dropout_rate": 0.}

        for param in changeable_params.keys():
            self.opt[param] = self.opt.get(param, changeable_params[param])
        return

    def texts2vec(self, sentences):
        """
        Convert texts to vector representations using embedder and padding up to self.opt["text_size"] tokens
        Args:
            sentences: list of lists of tokens

        Returns:
            array of embedded texts
        """
        pad = np.zeros(self.opt['embedding_size'])

        embeddings_batch = self.fasttext_model([sen[:self.opt['text_size']] for sen in sentences])
        embeddings_batch = [[pad] * (self.opt['text_size'] - len(tokens)) + tokens for tokens in embeddings_batch]

        embeddings_batch = np.asarray(embeddings_batch)
        return embeddings_batch

    def train_on_batch(self, texts, labels):
        """
        Train the model on the given batch
        Args:
            texts - list of texts (or list of lists of text tokens)
            labels - list of labels

        Returns:
            loss and metrics values on the given batch
        """
        if isinstance(texts[0], str):
            texts = self.tokenizer(list(texts))
        features = self.texts2vec(texts)
        onehot_labels = labels2onehot(labels, classes=self.classes)
        metrics_values = self.model.train_on_batch(features, onehot_labels)
        return metrics_values

    def infer_on_batch(self, texts, labels=None):
        """
        Infer the model on the given batch
        Args:
            texts - list of texts (or list of lists of text tokens)
            labels - list of labels

        Returns:
            loss and metrics values on the given batch, if labels are given
            predictions, otherwise
        """
        if isinstance(texts[0], str):
            texts = self.tokenizer(list(texts))
        if labels:
            features = self.texts2vec(texts)
            onehot_labels = labels2onehot(labels, classes=self.classes)
            metrics_values = self.model.test_on_batch(features, onehot_labels)
            return metrics_values
        else:
            features = self.texts2vec(texts)
            predictions = self.model.predict(features)
            return predictions

    def __call__(self, data, *args):
        """
        Infer on the given data
        Args:
            data: [list of sentences]
            *args:

        Returns:
            for each sentence:
                vector of probabilities to belong with each class
                or list of labels sentence belongs with
        """
        preds = np.array(self.infer_on_batch(data))

        labels = proba2labels(preds, confident_threshold=self.opt['confident_threshold'], classes=self.classes)
        return labels, [dict(zip(self.classes, preds[i])) for i in range(preds.shape[0])]

    def reset(self):
        pass

    def cnn_model(self, params):
        """
        Build un-compiled model of shallow-and-wide CNN
        Args:
            params: dictionary of parameters for NN

        Returns:
            Un-compiled model
        """

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        outputs = []
        for i in range(len(params['kernel_sizes_cnn'])):
            output_i = Conv1D(params['filters_cnn'], kernel_size=params['kernel_sizes_cnn'][i],
                              activation=None,
                              kernel_regularizer=l2(params['coef_reg_cnn']),
                              padding='same')(inp)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i = GlobalMaxPooling1D()(output_i)
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)

        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def dcnn_model(self, params):
        """
        Build un-compiled model of deep CNN
        Args:
            params: dictionary of parameters for NN

        Returns:
            Un-compiled model
        """

        if type(self.opt['filters_cnn']) is str:
            self.opt['filters_cnn'] = list(map(int, self.opt['filters_cnn'].split()))

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        output = inp

        for i in range(len(params['kernel_sizes_cnn'])):
            output = Conv1D(params['filters_cnn'][i], kernel_size=params['kernel_sizes_cnn'][i],
                            activation=None,
                            kernel_regularizer=l2(params['coef_reg_cnn']),
                            padding='same')(output)
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            output = MaxPooling1D()(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def cnn_model_max_and_aver_pool(self, params):
        """
        Build un-compiled model of shallow-and-wide CNN
        where average pooling after convolutions
        is replaced with concatenation of average and max poolings
        Args:
            params: dictionary of parameters for NN

        Returns:
            Un-compiled model
        """

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        outputs = []
        for i in range(len(params['kernel_sizes_cnn'])):
            output_i = Conv1D(params['filters_cnn'], kernel_size=params['kernel_sizes_cnn'][i],
                              activation=None,
                              kernel_regularizer=l2(params['coef_reg_cnn']),
                              padding='same')(inp)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i_0 = GlobalMaxPooling1D()(output_i)
            output_i_1 = GlobalAveragePooling1D()(output_i)
            output_i = Concatenate()([output_i_0, output_i_1])
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)

        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = BatchNormalization()(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def bilstm_model(self, params):
        """
        Build un-compiled BiLSTM
        Args:
            params: dictionary of parameters for NN

        Returns:
            Un-compiled model
        """

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        output = Bidirectional(LSTM(params['units_lstm'], activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(params['coef_reg_lstm']),
                                    dropout=params['dropout_rate'],
                                    recurrent_dropout=params['rec_dropout_rate']))(inp)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def bilstm_bilstm_model(self, params):
        """
        Build un-compiled two-layers BiLSTM
        Args:
            params: dictionary of parameters for NN

        Returns:
            Un-compiled model
        """

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        output = Bidirectional(LSTM(params['units_lstm_1'], activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(params['coef_reg_lstm']),
                                    dropout=params['dropout_rate'],
                                    recurrent_dropout=params['rec_dropout_rate']))(inp)

        output = Dropout(rate=params['dropout_rate'])(output)

        output = Bidirectional(LSTM(params['units_lstm_2'], activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(params['coef_reg_lstm']),
                                    dropout=params['dropout_rate'],
                                    recurrent_dropout=params['rec_dropout_rate']))(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def bilstm_cnn_model(self, params):
        """
        Build un-compiled BiLSTM-CNN
        Args:
            params: dictionary of parameters for NN

        Returns:
            Un-compiled model
        """

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        output = Bidirectional(LSTM(params['units_lstm'], activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(params['coef_reg_lstm']),
                                    dropout=params['dropout_rate'],
                                    recurrent_dropout=params['rec_dropout_rate']))(inp)

        output = Reshape(target_shape=(params['text_size'], 2 * params['units_lstm']))(output)
        outputs = []
        for i in range(len(params['kernel_sizes_cnn'])):
            output_i = Conv1D(params['filters_cnn'],
                              kernel_size=params['kernel_sizes_cnn'][i],
                              activation=None,
                              kernel_regularizer=l2(params['coef_reg_cnn']),
                              padding='same')(output)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i = GlobalMaxPooling1D()(output_i)
            outputs.append(output_i)

        output = Concatenate(axis=1)(outputs)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def cnn_bilstm_model(self, params):
        """
        Build un-compiled BiLSTM-CNN
        Args:
            params: dictionary of parameters for NN

        Returns:
            Un-compiled model
        """

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        outputs = []
        for i in range(len(params['kernel_sizes_cnn'])):
            output_i = Conv1D(params['filters_cnn'], kernel_size=params['kernel_sizes_cnn'][i],
                              activation=None,
                              kernel_regularizer=l2(params['coef_reg_cnn']),
                              padding='same')(inp)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i = MaxPooling1D()(output_i)
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)
        output = Dropout(rate=params['dropout_rate'])(output)

        output = Bidirectional(LSTM(params['units_lstm'], activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(params['coef_reg_lstm']),
                                    dropout=params['dropout_rate'],
                                    recurrent_dropout=params['rec_dropout_rate']))(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def bilstm_self_add_attention_model(self, params):
        """
        Method builds uncompiled model of BiLSTM with self additive attention
        Args:
            params: disctionary of parameters for NN

        Returns:
            Uncompiled model
        """

        inp = Input(shape=(params['text_size'], params['embedding_size']))
        output = Bidirectional(LSTM(params['units_lstm'], activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(params['coef_reg_lstm']),
                                    dropout=params['dropout_rate'],
                                    recurrent_dropout=params['rec_dropout_rate']))(inp)

        output = MaxPooling1D(pool_size=2, strides=3)(output)

        output = additive_self_attention(output, n_hidden=params.get("self_att_hid"),
                                         n_output_features=params.get("self_att_out"))
        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def bilstm_self_mult_attention_model(self, params):
        """
        Method builds uncompiled model of BiLSTM with self multiplicative attention
        Args:
            params: disctionary of parameters for NN

        Returns:
            Uncompiled model
        """

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        output = Bidirectional(LSTM(params['units_lstm'], activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(params['coef_reg_lstm']),
                                    dropout=params['dropout_rate'],
                                    recurrent_dropout=params['rec_dropout_rate']))(inp)

        output = MaxPooling1D(pool_size=2, strides=3)(output)

        output = multiplicative_self_attention(output, n_hidden=params.get("self_att_hid"),
                                               n_output_features=params.get("self_att_out"))
        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def bigru_model(self, params):
        """
        Method builds uncompiled model BiGRU
        Args:
            params: disctionary of parameters for NN

        Returns:
            Uncompiled model
        """

        inp = Input(shape=(params['text_size'], params['embedding_size']))

        output = Bidirectional(GRU(params['units_lstm'], activation='tanh',
                                   return_sequences=True,
                                   kernel_regularizer=l2(params['coef_reg_lstm']),
                                   dropout=params['dropout_rate'],
                                   recurrent_dropout=params['rec_dropout_rate']))(inp)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(params['dense_size'], activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=params['dropout_rate'])(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(params['coef_reg_den']))(output)
        act_output = Activation('sigmoid')(output)
        model = Model(inputs=inp, outputs=act_output)
        return model
