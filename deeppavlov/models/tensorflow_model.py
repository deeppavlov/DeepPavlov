"""
Inherit from this model to implement a `scikit-learn <http://scikit-learn.org/stable/>`_ model.
"""
import _pickle
from overrides import overrides
import tensorflow as tf
from tensorflow.contrib.training import HParams

from deeppavlov.models.model import Model


class TensorflowModel(Model):
    # def __init__(self, vocab, hps: HParams):
    #     self.vocab = vocab
    #     self._hps = hps

    def _add_placeholders(self):
        pass

    def _add_train_op(self):
        pass

    def build_graph(self):
        self._add_placeholders()
        self._add_seq2seq()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        if self._hps.mode == 'train':
            self._add_train_op()
        self._summaries = tf.summary.merge_all()

    @overrides
    def infer(self, data):
        """
        Load model and predict data.
        :param data: any type of input data
        :return: predicted values
        """
        pass

    # def _save(self):
    #     """
    #     Save model to file.
    #     """
    #     # TODO create ser_dir, not 'any_dir'.
    #     with open('any_dir') as f:
    #         pickle.dump(self._estimator, f)
    #
    # @overrides
    # def _load(self):
    #     """
    #     Load model from file.
    #     """
    #     # TODO decide what dir to check for loading.
    #     with open('any_dir') as f:
    #         _pickle.dump(self._estimator, f)

    def _gen_features(self, dataset):
        """
        Generate necessary for training features. Use :attr:`models`to generate input feature
        vector. The function should return an input feature vector.
        :param dataset: a dataset used for training
        """
        # TODO should return target vector only for train.
        raise NotImplementedError

    def forward(self, dataset):
        pass
