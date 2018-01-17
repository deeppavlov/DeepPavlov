from pathlib import Path
import inspect

from typing import Dict, Type
import numpy as np
from keras.layers import Dense, Input, concatenate, Activation
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.models import Model
from keras.regularizers import l2

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common import paths
from deeppavlov.core.common.attributes import check_attr_true
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.keras_model import KerasModel
from deeppavlov.models.classifiers.intents import metrics as metrics_file
from deeppavlov.models.classifiers.intents.utils import labels2onehot, log_metrics, proba2labels
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder
from deeppavlov.models.classifiers.intents.utils import md5_hashsum


@register('intent_model')
class KerasIntentModel(KerasModel):
    def __init__(self,
                 opt: Dict,
                 embedder: Type = FasttextEmbedder,
                 *args, **kwargs):
        """
        Method initializes model using parameters from opt
        Args:
            opt: model parameters
            *args:
            **kwargs:
        """
        super().__init__(opt, *args, **kwargs)

        try:
            classes_file = self.opt['classes_file']
        except KeyError:  # if no classes path is passed in json
            classes_file = Path(paths.USR_PATH).joinpath('intents', 'classes.txt')

        try:
            with open(str(classes_file)) as fin:
                self.classes = np.array(fin.read().split("\n"))

        except FileNotFoundError:
            print("Something is bad with the path to dataset classes file. "
                  "Provide the file path explicitly in json config.")
            raise

        self.n_classes = self.classes.shape[0]
        self.confident_threshold = self.opt['confident_threshold']
        if 'add_metrics' in self.opt.keys():
            self.add_metrics = self.opt['add_metrics'].split()
            self.add_metrics_values = len(self.add_metrics) * [0.]
        else:
            self.add_metrics = None

        self.fasttext_model = embedder
        self.opt['embedding_size'] = self.fasttext_model.dim
        current_fasttext_md5 = md5_hashsum([self.fasttext_model.model_path])

        params = {"model_name": self.opt['model_name'],
                  "optimizer_name": self.opt['optimizer'],
                  "lr": self.opt['lear_rate'],
                  "decay": self.opt['lear_rate_decay'],
                  "loss_name": self.opt['loss'],
                  "metrics_names": self.opt['lear_metrics'],
                  "add_metrics_file": metrics_file}

        self.model = self.load(**params, fname=self.model_path_)

        # Check if md5 hash sum of current loaded fasttext model
        # is equal to saved
        try:
            self.opt['fasttext_md5']
        except KeyError:
            self.opt['fasttext_md5'] = current_fasttext_md5
        else:
            if self.opt['fasttext_md5'] != current_fasttext_md5:
                raise ConfigError("Given fasttext model does NOT match fasttext model used previously to train loaded model")

        # List of parameters that could be changed
        # when the model is initialized from saved and is going to be trained further
        changeable_params = ["classes_file",
                             "lear_metrics",
                             "confident_threshold",
                             "optimizer",
                             "lear_rate",
                             "lear_rate_decay",
                             "loss",
                             "coef_reg_cnn",
                             "coef_reg_den",
                             "dropout_rate",
                             "epochs",
                             "batch_size",
                             "val_every_n_epochs",
                             "verbose",
                             "val_patience"]
        # Reinitializing of parameters
        for param in changeable_params:
            self.opt[param] = opt[param]

        self.metrics_names = self.model.metrics_names
        self.metrics_values = len(self.metrics_names) * [0.]

    def texts2vec(self, sentences):
        embeddings_batch = []
        for sen in sentences:
            tokens = [el for el in sen.split() if el]
            if len(tokens) > self.opt['text_size']:
                tokens = tokens[:self.opt['text_size']]

            embeddings = self.fasttext_model.infer(' '.join(tokens))
            if len(tokens) < self.opt['text_size']:
                pads = [np.zeros(self.opt['embedding_size'])
                        for _ in range(self.opt['text_size'] - len(tokens))]
                embeddings = pads + embeddings

            embeddings = np.asarray(embeddings)
            embeddings_batch.append(embeddings)

        embeddings_batch = np.asarray(embeddings_batch)
        return embeddings_batch

    def train_on_batch(self, batch):
        """
        Method trains the model on the given batch
        Args:
            batch - list of data where batch[0] is list of texts and batch[1] is list of labels

        Returns:
            loss and metrics values on the given batch
        """
        texts = list(batch[0])
        labels = list(batch[1])
        features = self.texts2vec(texts)
        onehot_labels = labels2onehot(labels, classes=self.classes)
        metrics_values = self.model.train_on_batch(features, onehot_labels)
        return metrics_values

    def infer_on_batch(self, batch):
        """
        Method infers the model on the given batch
        Args:
            batch - list of data where batch[0] is list of texts (and if given, batch[1] is list of labels)

        Returns:
            loss and metrics values on the given batch, if labels are given
            predictions, otherwise
        """
        if self.opt['model_name']:
            return self.infer_on_batch_tfidf(batch)
        texts = list(batch[0])
        if len(batch) == 2:
            labels = list(batch[1])
            features = self.texts2vec(texts)
            onehot_labels = labels2onehot(labels, classes=self.classes)
            metrics_values = self.model.test_on_batch(features, onehot_labels)
            return metrics_values
        else:
            features = self.texts2vec(texts)
            predictions = self.model.predict(features)
            return predictions

    @check_attr_true('train_now')
    def train(self, dataset, *args, **kwargs):
        """
        Method trains the model using batches and validation
        Args:
            dataset: instance of class Dataset

        Returns: None

        """
        updates = 0
        val_loss = 1e100
        val_increase = 0
        epochs_done = 0

        n_train_samples = len(dataset.data['train'])

        valid_iter_all = dataset.iter_all(data_type='valid')
        valid_x = []
        valid_y = []
        for valid_i, valid_sample in enumerate(valid_iter_all):
            valid_x.append(valid_sample[0])
            valid_y.append(valid_sample[1])

        valid_x = self.texts2vec(valid_x)
        valid_y = labels2onehot(valid_y, classes=self.classes)

        print('\n____Training over {} samples____\n\n'.format(n_train_samples))

        while epochs_done < self.opt['epochs']:
            batch_gen = dataset.batch_generator(batch_size=self.opt['batch_size'],
                                                data_type='train')
            for step, batch in enumerate(batch_gen):
                metrics_values = self.train_on_batch(batch)
                updates += 1

                if self.opt['verbose'] and step % 50 == 0:
                    log_metrics(names=self.metrics_names,
                                values=metrics_values,
                                updates=updates,
                                mode='train')

            epochs_done += 1
            if epochs_done % self.opt['val_every_n_epochs'] == 0:
                if 'valid' in dataset.data.keys():
                    valid_metrics_values = self.model.test_on_batch(x=valid_x, y=valid_y)

                    log_metrics(names=self.metrics_names,
                                values=valid_metrics_values,
                                mode='valid')
                    if valid_metrics_values[0] > val_loss:
                        val_increase += 1
                        print("__Validation impatience {} out of {}".format(
                            val_increase, self.opt['val_patience']))
                        if val_increase == self.opt['val_patience']:
                            print("___Stop training: validation is out of patience___")
                            break
                    else:
                        val_increase = 0
                        val_loss = valid_metrics_values[0]
            print('epochs_done: {}'.format(epochs_done))

        self.save()

    def infer(self, data, return_proba=False, *args):
        """
        Method returns predictions on the given data
        Args:
            data: sentence or list of sentences
            *args:

        Returns:
            Predictions for the given data
        """
        if type(data) is str:
            features = self.texts2vec([data])
            preds = self.model.predict_on_batch(features)[0]
            if return_proba:
                return preds
            else:
                return proba2labels([preds], confident_threshold=self.confident_threshold, classes=self.classes)[0]

        elif inspect.isgeneratorfunction(data):
            preds = []
            for step, batch in enumerate(data):
                preds.extend(self.infer_on_batch(batch))
                preds = np.array(preds)
        elif type(data) is list:
            features = self.texts2vec(data)
            preds = self.model.predict_on_batch(features)
        else:
            raise ConfigError("Not understand data type for inference")

        if return_proba:
            return preds
        else:
            return proba2labels(preds, confident_threshold=self.confident_threshold, classes=self.classes)


    def cnn_model(self, params):
        """
        Method builds uncompiled model of shallow-and-wide CNN
        Args:
            params: disctionary of parameters for NN

        Returns:
            Uncompiled model
        """
        if type(self.opt['kernel_sizes_cnn']) is str:
            self.opt['kernel_sizes_cnn'] = list(map(int, self.opt['kernel_sizes_cnn'].split()))

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
        Method builds uncompiled model of deep CNN
        Args:
            params: dictionary of parameters for NN

        Returns:
            Uncompiled model
        """
        if type(self.opt['kernel_sizes_cnn']) is str:
            self.opt['kernel_sizes_cnn'] = list(map(int, self.opt['kernel_sizes_cnn'].split()))

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

    def reset(self):
        pass
