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

from typing import List, Tuple, Optional, Generator, Union
from pathlib import Path
from typing import List, Tuple, Optional
from copy import deepcopy

import numpy as np
import keras.metrics
import keras.optimizers
from keras import backend as K
from keras.layers import Dense, Input
from keras.layers import concatenate, Activation, Concatenate, Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.regularizers import l2
from overrides import overrides

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.file import save_json, read_json
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.keras_model import KerasModel
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.layers.keras_layers import additive_self_attention, multiplicative_self_attention


log = get_logger(__name__)


@register('keras_classification_model')
class KerasClassificationModel(KerasModel):
    """
    Class implements Keras model for classification task for multi-class multi-labeled data.

    Args:
        text_size: maximal length of text in tokens (words),
                longer texts are cutted,
                shorter ones are padded by zeros (pre-padding)
        embedding_size: embedding_size from embedder in pipeline
        n_classes: number of considered classes
        model_name: particular method of this class to initialize model configuration
        optimizer: function name from keras.optimizers
        loss: function name from keras.losses.
        learning_rate: learning rate for optimizer.
        learning_rate_decay: learning rate decay for optimizer
        last_layer_activation: parameter that determines activation function after classification layer.
                For multi-label classification use `sigmoid`,
                otherwise, `softmax`.
        restore_lr: in case of loading pre-trained model \
                whether to init learning rate with the final learning rate value from saved opt
        classes: list or generator of considered classes

    Attributes:
        opt: dictionary with all model parameters
        n_classes: number of considered classes
        model: keras model itself
        epochs_done: number of epochs that were done
        batches_seen: number of epochs that were seen
        train_examples_seen: number of training samples that were seen
        sess: tf session
        optimizer: keras.optimizers instance
        classes: list of considered classes
    """

    def __init__(self, text_size: int, embedding_size: int, n_classes: int,
                 model_name: str, optimizer: str = "Adam", loss: str = "binary_crossentropy",
                 learning_rate: float = 0.01, learning_rate_decay: float = 0.,
                 last_layer_activation="sigmoid",
                 restore_lr: bool = False,
                 classes: Optional[Union[list, Generator]] = None,
                 **kwargs):
        """
        Initialize model using parameters
        from opt dictionary (from config), if model is being initialized from saved.
        """
        if classes is not None:
            classes = list(classes)

        given_opt = {"text_size": text_size,
                     "embedding_size": embedding_size,
                     "n_classes": n_classes,
                     "model_name": model_name,
                     "optimizer": optimizer,
                     "loss": loss,
                     "learning_rate": learning_rate,
                     "learning_rate_decay": learning_rate_decay,
                     "last_layer_activation": last_layer_activation,
                     "restore_lr": restore_lr,
                     "classes": classes,
                     **kwargs}
        self.opt = deepcopy(given_opt)
        self.model = None

        super().__init__(**given_opt)

        if classes is not None:
            self.classes = self.opt.get("classes")

        self.n_classes = self.opt.get('n_classes')
        if self.n_classes == 0:
            raise ConfigError("Please, provide vocabulary with considered intents.")

        self.load(model_name=model_name)
        # in case of pre-trained after loading in self.opt we have stored parameters
        # now we can restore lear rate if needed
        if restore_lr:
            learning_rate = self.opt.get("final_learning_rate", learning_rate)

        self.model = self.compile(self.model, optimizer_name=optimizer, loss_name=loss,
                                  learning_rate=learning_rate, learning_rate_decay=learning_rate_decay)

        self._change_not_fixed_params(**given_opt)

        summary = ['Model was successfully initialized!', 'Model summary:']
        self.model.summary(print_fn=summary.append)
        log.info('\n'.join(summary))

    def _change_not_fixed_params(self, **kwargs) -> None:
        """
        Change changable parameters from saved model to given ones.

        Args:
            kwargs: dictionary of new parameters

        Returns:
            None
        """
        fixed_params = [
            "n_classes",
            "model_name",
            "embedding_size",
            "fasttext_md5",
            "kernel_sizes_cnn",
            "filters_cnn",
            "dense_size",
            "units_lstm",
            "units_lstm_1",
            "units_lstm_2",
            "self_att_hid",
            "self_att_out"
        ]
        for param in self.opt.keys():
            if param not in fixed_params:
                self.opt[param] = kwargs.get(param)
        return

    def pad_texts(self, sentences: List[List[np.ndarray]]) -> np.ndarray:
        """
        Cut and pad tokenized texts to self.opt["text_size"] tokens

        Args:
            sentences: list of lists of tokens

        Returns:
            array of embedded texts
        """
        pad = np.zeros(self.opt['embedding_size'])
        cutted_batch = [sen[:self.opt['text_size']] for sen in sentences]
        cutted_batch = [[pad] * (self.opt['text_size'] - len(tokens)) + list(tokens) for tokens in cutted_batch]
        return np.asarray(cutted_batch)

    def train_on_batch(self, texts: List[List[np.ndarray]], labels: list) -> [float, List[float]]:
        """
        Train the model on the given batch

        Args:
            texts: list of tokenized text samples
            labels: list of labels

        Returns:
            metrics values on the given batch
        """
        features = self.pad_texts(texts)
        metrics_values = self.model.train_on_batch(features, np.squeeze(np.array(labels)))
        return metrics_values

    def infer_on_batch(self, texts: List[List[np.ndarray]], labels: list = None) -> [float, List[float], np.ndarray]:
        """
        Infer the model on the given batch

        Args:
            texts: list of tokenized text samples
            labels: list of labels

        Returns:
            metrics values on the given batch, if labels are given
            predictions, otherwise
        """
        if labels:
            features = self.pad_texts(texts)
            metrics_values = self.model.test_on_batch(features, np.squeeze(np.array(labels)))
            return metrics_values
        else:
            features = self.pad_texts(texts)
            predictions = self.model.predict(features)
            return predictions

    def __call__(self, data: List[List[np.ndarray]], *args) -> List[List[float]]:
        """
        Infer on the given data

        Args:
            data: list of tokenized text samples
            *args: additional arguments

        Returns:
            for each sentence:
                vector of probabilities to belong with each class
                or list of labels sentence belongs with
        """
        preds = np.array(self.infer_on_batch(data), dtype="float64").tolist()
        return preds

    def init_model_from_scratch(self, model_name: str) -> Model:
        """
        Initialize uncompiled model from scratch with given params

        Args:
            model_name: name of model function described as a method of this class

        Returns:
            compiled model with given network and learning parameters
        """
        log.info(f'[initializing `{self.__class__.__name__}` from scratch as {model_name}]')
        model_func = getattr(self, model_name, None)
        if callable(model_func):
            model = model_func(**self.opt)
        else:
            raise AttributeError("Model {} is not defined".format(model_name))

        return model

    @overrides
    def load(self, model_name: str) -> None:
        """
        Initialize uncompiled model from saved params and weights

        Args:
            model_name: name of model function described as a method of this class

        Returns:
            model with loaded weights and network parameters from files
            but compiled with given learning parameters
        """
        if self.load_path:
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            opt_path = Path("{}_opt.json".format(str(self.load_path.resolve())))
            weights_path = Path("{}.h5".format(str(self.load_path.resolve())))

            if opt_path.exists() and weights_path.exists():

                log.info("[initializing `{}` from saved]".format(self.__class__.__name__))

                self.opt = read_json(opt_path)

                model_func = getattr(self, model_name, None)
                if callable(model_func):
                    model = model_func(**self.opt)
                else:
                    raise AttributeError("Model {} is not defined".format(model_name))

                log.info("[loading weights from {}]".format(weights_path.name))
                model.load_weights(str(weights_path))

                self.model = model

                return None
            else:
                self.model = self.init_model_from_scratch(model_name)
                return None
        else:
            log.warning("No `load_path` is provided for {}".format(self.__class__.__name__))
            self.model = self.init_model_from_scratch(model_name)
            return None

    def compile(self, model: Model, optimizer_name: str, loss_name: str,
                learning_rate: float = 0.01, learning_rate_decay: float = 0.) -> Model:
        """
        Compile model with given optimizer and loss

        Args:
            model: keras uncompiled model
            optimizer_name: name of optimizer from keras.optimizers
            loss_name: loss function name (from keras.losses)
            learning_rate: learning rate.
            learning_rate_decay: learning rate decay.

        Returns:

        """
        optimizer_func = getattr(keras.optimizers, optimizer_name, None)
        if callable(optimizer_func):
            if not (learning_rate is None):
                if not (learning_rate_decay is None):
                    self.optimizer = optimizer_func(lr=learning_rate, decay=learning_rate_decay)
                else:
                    self.optimizer = optimizer_func(lr=learning_rate)
            elif not (learning_rate_decay is None):
                self.optimizer = optimizer_func(decay=learning_rate_decay)
            else:
                self.optimizer = optimizer_func()
        else:
            raise AttributeError("Optimizer {} is not defined in `keras.optimizers`".format(optimizer_name))

        loss_func = getattr(keras.losses, loss_name, None)
        if callable(loss_func):
            loss = loss_func
        else:
            raise AttributeError("Loss {} is not defined".format(loss_name))

        model.compile(optimizer=self.optimizer,
                      loss=loss)
        return model

    @overrides
    def save(self, fname: str = None) -> None:
        """
        Save the model parameters into <<fname>>_opt.json (or <<ser_file>>_opt.json)
        and model weights into <<fname>>.h5 (or <<ser_file>>.h5)
        Args:
            fname: file_path to save model. If not explicitly given seld.opt["ser_file"] will be used

        Returns:
            None
        """
        if not fname:
            fname = self.save_path
        else:
            fname = Path(fname).resolve()

        if not fname.parent.is_dir():
            raise ConfigError("Provided save path is incorrect!")
        else:
            opt_path = f"{fname}_opt.json"
            weights_path = f"{fname}.h5"
            log.info(f"[saving model to {opt_path}]")
            self.model.save_weights(weights_path)

        # if model was loaded from one path and saved to another one
        # then change load_path to save_path for config
        self.opt["epochs_done"] = self.epochs_done
        self.opt["final_learning_rate"] = K.eval(self.optimizer.lr) / (1. +
                                                                       K.eval(self.optimizer.decay) * self.batches_seen)

        if self.opt.get("load_path") and self.opt.get("save_path"):
            if self.opt.get("save_path") != self.opt.get("load_path"):
                self.opt["load_path"] = str(self.opt["save_path"])
        save_json(self.opt, opt_path)

    def cnn_model(self, kernel_sizes_cnn: List[int], filters_cnn: int, dense_size: int,
                  coef_reg_cnn: float = 0., coef_reg_den: float = 0., dropout_rate: float = 0.,
                  input_projection_size: Optional[int] = None, **kwargs) -> Model:
        """
        Build un-compiled model of shallow-and-wide CNN.

        Args:
            kernel_sizes_cnn: list of kernel sizes of convolutions.
            filters_cnn: number of filters for convolutions.
            dense_size: number of units for dense layer.
            coef_reg_cnn: l2-regularization coefficient for convolutions.
            coef_reg_den: l2-regularization coefficient for dense layers.
            dropout_rate: dropout rate used after convolutions and between dense layers.
            input_projection_size: if not None, adds Dense layer (with ``relu`` activation)
                                   right after input layer to the size ``input_projection_size``.
                                   Useful for input dimentionaliry recuction. Default: ``None``.
            kwargs: other non-used parameters

        Returns:
            keras.models.Model: uncompiled instance of Keras Model
        """
        inp = Input(shape=(self.opt['text_size'], self.opt['embedding_size']))
        output = inp

        if input_projection_size is not None:
            output = Dense(input_projection_size, activation='relu')(output)

        outputs = []
        for i in range(len(kernel_sizes_cnn)):
            output_i = Conv1D(filters_cnn, kernel_size=kernel_sizes_cnn[i],
                              activation=None,
                              kernel_regularizer=l2(coef_reg_cnn),
                              padding='same')(output)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i = GlobalMaxPooling1D()(output_i)
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)

        output = Dropout(rate=dropout_rate)(output)
        output = Dense(dense_size, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        output = BatchNormalization()(output)
        act_output = Activation(self.opt.get("last_layer_activation", "sigmoid"))(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def dcnn_model(self, kernel_sizes_cnn: List[int], filters_cnn: List[int], dense_size: int,
                   coef_reg_cnn: float = 0., coef_reg_den: float = 0., dropout_rate: float = 0.,
                   input_projection_size: Optional[int] = None, **kwargs) -> Model:
        """
        Build un-compiled model of deep CNN.

        Args:
            kernel_sizes_cnn: list of kernel sizes of convolutions.
            filters_cnn: number of filters for convolutions.
            dense_size: number of units for dense layer.
            coef_reg_cnn: l2-regularization coefficient for convolutions.
            coef_reg_den: l2-regularization coefficient for dense layers.
            dropout_rate: dropout rate used after convolutions and between dense layers.
            input_projection_size: if not None, adds Dense layer (with ``relu`` activation)
                                   right after input layer to the size ``input_projection_size``.
                                   Useful for input dimentionaliry recuction. Default: ``None``.
            kwargs: other non-used parameters

        Returns:
            keras.models.Model: uncompiled instance of Keras Model
        """
        inp = Input(shape=(self.opt['text_size'], self.opt['embedding_size']))
        output = inp

        if input_projection_size is not None:
            output = Dense(input_projection_size, activation='relu')(output)

        for i in range(len(kernel_sizes_cnn)):
            output = Conv1D(filters_cnn[i], kernel_size=kernel_sizes_cnn[i],
                            activation=None,
                            kernel_regularizer=l2(coef_reg_cnn),
                            padding='same')(output)
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            output = MaxPooling1D()(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(dense_size, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        output = BatchNormalization()(output)
        act_output = Activation(self.opt.get("last_layer_activation", "sigmoid"))(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def cnn_model_max_and_aver_pool(self, kernel_sizes_cnn: List[int], filters_cnn: int, dense_size: int,
                                    coef_reg_cnn: float = 0., coef_reg_den: float = 0., dropout_rate: float = 0.,
                                    input_projection_size: Optional[int] = None, **kwargs) -> Model:
        """
        Build un-compiled model of shallow-and-wide CNN where average pooling after convolutions is replaced with
        concatenation of average and max poolings.

        Args:
            kernel_sizes_cnn: list of kernel sizes of convolutions.
            filters_cnn: number of filters for convolutions.
            dense_size: number of units for dense layer.
            coef_reg_cnn: l2-regularization coefficient for convolutions. Default: ``0.0``.
            coef_reg_den: l2-regularization coefficient for dense layers. Default: ``0.0``.
            dropout_rate: dropout rate used after convolutions and between dense layers. Default: ``0.0``.
            input_projection_size: if not None, adds Dense layer (with ``relu`` activation)
                                   right after input layer to the size ``input_projection_size``.
                                   Useful for input dimentionaliry recuction. Default: ``None``.
            kwargs: other non-used parameters

        Returns:
            keras.models.Model: uncompiled instance of Keras Model
        """

        inp = Input(shape=(self.opt['text_size'], self.opt['embedding_size']))
        output = inp

        if input_projection_size is not None:
            output = Dense(input_projection_size, activation='relu')(output)

        outputs = []
        for i in range(len(kernel_sizes_cnn)):
            output_i = Conv1D(filters_cnn, kernel_size=kernel_sizes_cnn[i],
                              activation=None,
                              kernel_regularizer=l2(coef_reg_cnn),
                              padding='same')(output)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i_0 = GlobalMaxPooling1D()(output_i)
            output_i_1 = GlobalAveragePooling1D()(output_i)
            output_i = Concatenate()([output_i_0, output_i_1])
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)

        output = Dropout(rate=dropout_rate)(output)
        output = Dense(dense_size, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        output = BatchNormalization()(output)
        act_output = Activation(self.opt.get("last_layer_activation", "sigmoid"))(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def bilstm_model(self, units_lstm: int, dense_size: int,
                     coef_reg_lstm: float = 0., coef_reg_den: float = 0.,
                     dropout_rate: float = 0., rec_dropout_rate: float = 0.,
                     input_projection_size: Optional[int] = None, **kwargs) -> Model:
        """
        Build un-compiled BiLSTM.

        Args:
            units_lstm (int): number of units for LSTM.
            dense_size (int): number of units for dense layer.
            coef_reg_lstm (float): l2-regularization coefficient for LSTM. Default: ``0.0``.
            coef_reg_den (float): l2-regularization coefficient for dense layers. Default: ``0.0``.
            dropout_rate (float): dropout rate to be used after BiLSTM and between dense layers. Default: ``0.0``.
            rec_dropout_rate (float): dropout rate for LSTM. Default: ``0.0``.
            input_projection_size: if not None, adds Dense layer (with ``relu`` activation)
                                   right after input layer to the size ``input_projection_size``.
                                   Useful for input dimentionaliry recuction. Default: ``None``.
            kwargs: other non-used parameters

        Returns:
            keras.models.Model: uncompiled instance of Keras Model
        """

        inp = Input(shape=(self.opt['text_size'], self.opt['embedding_size']))
        output = inp

        if input_projection_size is not None:
            output = Dense(input_projection_size, activation='relu')(output)

        output = Bidirectional(LSTM(units_lstm, activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(coef_reg_lstm),
                                    dropout=dropout_rate,
                                    recurrent_dropout=rec_dropout_rate))(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(dense_size, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        act_output = Activation(self.opt.get("last_layer_activation", "sigmoid"))(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def bilstm_bilstm_model(self, units_lstm_1: int, units_lstm_2: int, dense_size: int,
                            coef_reg_lstm: float = 0., coef_reg_den: float = 0.,
                            dropout_rate: float = 0., rec_dropout_rate: float = 0.,
                            input_projection_size: Optional[int] = None, **kwargs) -> Model:
        """
        Build un-compiled two-layers BiLSTM.

        Args:
            units_lstm_1: number of units for the first LSTM layer.
            units_lstm_2: number of units for the second LSTM layer.
            dense_size: number of units for dense layer.
            coef_reg_lstm: l2-regularization coefficient for LSTM. Default: ``0.0``.
            coef_reg_den: l2-regularization coefficient for dense layers. Default: ``0.0``.
            dropout_rate: dropout rate to be used after BiLSTM and between dense layers. Default: ``0.0``.
            rec_dropout_rate: dropout rate for LSTM. Default: ``0.0``.
            input_projection_size: if not None, adds Dense layer (with ``relu`` activation)
                                   right after input layer to the size ``input_projection_size``.
                                   Useful for input dimentionaliry recuction. Default: ``None``.
            kwargs: other non-used parameters

        Returns:
            keras.models.Model: uncompiled instance of Keras Model
        """

        inp = Input(shape=(self.opt['text_size'], self.opt['embedding_size']))
        output = inp

        if input_projection_size is not None:
            output = Dense(input_projection_size, activation='relu')(output)

        output = Bidirectional(LSTM(units_lstm_1, activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(coef_reg_lstm),
                                    dropout=dropout_rate,
                                    recurrent_dropout=rec_dropout_rate))(output)

        output = Dropout(rate=dropout_rate)(output)

        output = Bidirectional(LSTM(units_lstm_2, activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(coef_reg_lstm),
                                    dropout=dropout_rate,
                                    recurrent_dropout=rec_dropout_rate))(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(dense_size, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        act_output = Activation(self.opt.get("last_layer_activation", "sigmoid"))(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def bilstm_cnn_model(self, units_lstm: int, kernel_sizes_cnn: List[int], filters_cnn: int, dense_size: int,
                         coef_reg_lstm: float = 0., coef_reg_cnn: float = 0., coef_reg_den: float = 0.,
                         dropout_rate: float = 0., rec_dropout_rate: float = 0.,
                         input_projection_size: Optional[int] = None, **kwargs) -> Model:
        """
        Build un-compiled BiLSTM-CNN.

        Args:
            units_lstm: number of units for LSTM.
            kernel_sizes_cnn: list of kernel sizes of convolutions.
            filters_cnn: number of filters for convolutions.
            dense_size: number of units for dense layer.
            coef_reg_lstm: l2-regularization coefficient for LSTM. Default: ``0.0``.
            coef_reg_cnn: l2-regularization coefficient for convolutions. Default: ``0.0``.
            coef_reg_den: l2-regularization coefficient for dense layers. Default: ``0.0``.
            dropout_rate: dropout rate to be used after BiLSTM and between dense layers. Default: ``0.0``.
            rec_dropout_rate: dropout rate for LSTM. Default: ``0.0``.
            input_projection_size: if not None, adds Dense layer (with ``relu`` activation)
                                   right after input layer to the size ``input_projection_size``.
                                   Useful for input dimentionaliry recuction. Default: ``None``.
            kwargs: other non-used parameters

        Returns:
            keras.models.Model: uncompiled instance of Keras Model
        """

        inp = Input(shape=(self.opt['text_size'], self.opt['embedding_size']))
        output = inp

        if input_projection_size is not None:
            output = Dense(input_projection_size, activation='relu')(output)

        output = Bidirectional(LSTM(units_lstm, activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(coef_reg_lstm),
                                    dropout=dropout_rate,
                                    recurrent_dropout=rec_dropout_rate))(output)

        output = Reshape(target_shape=(self.opt['text_size'], 2 * units_lstm))(output)
        outputs = []
        for i in range(len(kernel_sizes_cnn)):
            output_i = Conv1D(filters_cnn,
                              kernel_size=kernel_sizes_cnn[i],
                              activation=None,
                              kernel_regularizer=l2(coef_reg_cnn),
                              padding='same')(output)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i = GlobalMaxPooling1D()(output_i)
            outputs.append(output_i)

        output = Concatenate(axis=1)(outputs)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(dense_size, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        act_output = Activation(self.opt.get("last_layer_activation", "sigmoid"))(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def cnn_bilstm_model(self, kernel_sizes_cnn: List[int], filters_cnn: int, units_lstm: int, dense_size: int,
                         coef_reg_cnn: float = 0., coef_reg_lstm: float = 0., coef_reg_den: float = 0.,
                         dropout_rate: float = 0., rec_dropout_rate: float = 0.,
                         input_projection_size: Optional[int] = None, **kwargs) -> Model:
        """
        Build un-compiled BiLSTM-CNN.

        Args:
            kernel_sizes_cnn: list of kernel sizes of convolutions.
            filters_cnn: number of filters for convolutions.
            units_lstm: number of units for LSTM.
            dense_size: number of units for dense layer.
            coef_reg_cnn: l2-regularization coefficient for convolutions. Default: ``0.0``.
            coef_reg_lstm: l2-regularization coefficient for LSTM. Default: ``0.0``.
            coef_reg_den: l2-regularization coefficient for dense layers. Default: ``0.0``.
            dropout_rate: dropout rate to be used after BiLSTM and between dense layers. Default: ``0.0``.
            rec_dropout_rate: dropout rate for LSTM. Default: ``0.0``.
            input_projection_size: if not None, adds Dense layer (with ``relu`` activation)
                                   right after input layer to the size ``input_projection_size``.
                                   Useful for input dimentionaliry recuction. Default: ``None``.
            kwargs: other non-used parameters

        Returns:
            keras.models.Model: uncompiled instance of Keras Model
        """

        inp = Input(shape=(self.opt['text_size'], self.opt['embedding_size']))
        output = inp

        if input_projection_size is not None:
            output = Dense(input_projection_size, activation='relu')(output)

        outputs = []
        for i in range(len(kernel_sizes_cnn)):
            output_i = Conv1D(filters_cnn, kernel_size=kernel_sizes_cnn[i],
                              activation=None,
                              kernel_regularizer=l2(coef_reg_cnn),
                              padding='same')(output)
            output_i = BatchNormalization()(output_i)
            output_i = Activation('relu')(output_i)
            output_i = MaxPooling1D()(output_i)
            outputs.append(output_i)

        output = concatenate(outputs, axis=1)
        output = Dropout(rate=dropout_rate)(output)

        output = Bidirectional(LSTM(units_lstm, activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(coef_reg_lstm),
                                    dropout=dropout_rate,
                                    recurrent_dropout=rec_dropout_rate))(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(dense_size, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        act_output = Activation(self.opt.get("last_layer_activation", "sigmoid"))(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def bilstm_self_add_attention_model(self, units_lstm: int, dense_size: int, self_att_hid: int, self_att_out: int,
                                        coef_reg_lstm: float = 0., coef_reg_den: float = 0.,
                                        dropout_rate: float = 0., rec_dropout_rate: float = 0.,
                                        input_projection_size: Optional[int] = None, **kwargs) -> Model:
        """
        Method builds uncompiled model of BiLSTM with self additive attention.

        Args:
            units_lstm: number of units for LSTM.
            self_att_hid: number of hidden units in self-attention
            self_att_out: number of output units in self-attention
            dense_size: number of units for dense layer.
            coef_reg_lstm: l2-regularization coefficient for LSTM. Default: ``0.0``.
            coef_reg_den: l2-regularization coefficient for dense layers. Default: ``0.0``.
            dropout_rate: dropout rate to be used after BiLSTM and between dense layers. Default: ``0.0``.
            rec_dropout_rate: dropout rate for LSTM. Default: ``0.0``.
            input_projection_size: if not None, adds Dense layer (with ``relu`` activation)
                                   right after input layer to the size ``input_projection_size``.
                                   Useful for input dimentionaliry recuction. Default: ``None``.
            kwargs: other non-used parameters

        Returns:
            keras.models.Model: uncompiled instance of Keras Model
        """

        inp = Input(shape=(self.opt['text_size'], self.opt['embedding_size']))
        output = inp

        if input_projection_size is not None:
            output = Dense(input_projection_size, activation='relu')(output)

        output = Bidirectional(LSTM(units_lstm, activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(coef_reg_lstm),
                                    dropout=dropout_rate,
                                    recurrent_dropout=rec_dropout_rate))(output)

        output = MaxPooling1D(pool_size=2, strides=3)(output)

        output = additive_self_attention(output, n_hidden=self_att_hid,
                                         n_output_features=self_att_out)
        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(dense_size, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        act_output = Activation(self.opt.get("last_layer_activation", "sigmoid"))(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def bilstm_self_mult_attention_model(self, units_lstm: int, dense_size: int, self_att_hid: int, self_att_out: int,
                                         coef_reg_lstm: float = 0., coef_reg_den: float = 0.,
                                         dropout_rate: float = 0., rec_dropout_rate: float = 0.,
                                         input_projection_size: Optional[int] = None, **kwargs) -> Model:
        """
        Method builds uncompiled model of BiLSTM with self multiplicative attention.

        Args:
            units_lstm: number of units for LSTM.
            self_att_hid: number of hidden units in self-attention
            self_att_out: number of output units in self-attention
            dense_size: number of units for dense layer.
            coef_reg_lstm: l2-regularization coefficient for LSTM. Default: ``0.0``.
            coef_reg_den: l2-regularization coefficient for dense layers. Default: ``0.0``.
            dropout_rate: dropout rate to be used after BiLSTM and between dense layers. Default: ``0.0``.
            rec_dropout_rate: dropout rate for LSTM. Default: ``0.0``.
            input_projection_size: if not None, adds Dense layer (with ``relu`` activation)
                                   right after input layer to the size ``input_projection_size``.
                                   Useful for input dimentionaliry recuction. Default: ``None``.
            kwargs: other non-used parameters

        Returns:
            keras.models.Model: uncompiled instance of Keras Model
        """

        inp = Input(shape=(self.opt['text_size'], self.opt['embedding_size']))
        output = inp

        if input_projection_size is not None:
            output = Dense(input_projection_size, activation='relu')(output)

        output = Bidirectional(LSTM(units_lstm, activation='tanh',
                                    return_sequences=True,
                                    kernel_regularizer=l2(coef_reg_lstm),
                                    dropout=dropout_rate,
                                    recurrent_dropout=rec_dropout_rate))(output)

        output = MaxPooling1D(pool_size=2, strides=3)(output)

        output = multiplicative_self_attention(output, n_hidden=self_att_hid,
                                               n_output_features=self_att_out)
        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(dense_size, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        act_output = Activation(self.opt.get("last_layer_activation", "sigmoid"))(output)
        model = Model(inputs=inp, outputs=act_output)
        return model

    def bigru_model(self, units_lstm: int, dense_size: int,
                    coef_reg_lstm: float = 0., coef_reg_den: float = 0.,
                    dropout_rate: float = 0., rec_dropout_rate: float = 0.,
                    input_projection_size: Optional[int] = None, **kwargs) -> Model:
        """
        Method builds uncompiled model BiGRU.

        Args:
            units_lstm: number of units for GRU.
            dense_size: number of units for dense layer.
            coef_reg_lstm: l2-regularization coefficient for GRU. Default: ``0.0``.
            coef_reg_den: l2-regularization coefficient for dense layers. Default: ``0.0``.
            dropout_rate: dropout rate to be used after BiGRU and between dense layers. Default: ``0.0``.
            rec_dropout_rate: dropout rate for GRU. Default: ``0.0``.
            input_projection_size: if not None, adds Dense layer (with ``relu`` activation)
                                   right after input layer to the size ``input_projection_size``.
                                   Useful for input dimentionaliry recuction. Default: ``None``.
            kwargs: other non-used parameters

        Returns:
            keras.models.Model: uncompiled instance of Keras Model
        """

        inp = Input(shape=(self.opt['text_size'], self.opt['embedding_size']))
        output = inp

        if input_projection_size is not None:
            output = Dense(input_projection_size, activation='relu')(output)

        output = Bidirectional(GRU(units_lstm, activation='tanh',
                                   return_sequences=True,
                                   kernel_regularizer=l2(coef_reg_lstm),
                                   dropout=dropout_rate,
                                   recurrent_dropout=rec_dropout_rate))(output)

        output = GlobalMaxPooling1D()(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(dense_size, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        output = Activation('relu')(output)
        output = Dropout(rate=dropout_rate)(output)
        output = Dense(self.n_classes, activation=None,
                       kernel_regularizer=l2(coef_reg_den))(output)
        act_output = Activation(self.opt.get("last_layer_activation", "sigmoid"))(output)
        model = Model(inputs=inp, outputs=act_output)
        return model
