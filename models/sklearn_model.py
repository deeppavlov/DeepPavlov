"""
Inherit from this model to implement a `scikit-learn <http://scikit-learn.org/stable/>`_ model.
"""
import pickle
from overrides import overrides

from models.model import Model


class SklearnModel(Model):
    def __init__(self, config_path, models, estimator):
        # TODO where the estimator parameters should initialize?
        self._estimator = estimator
        super(SklearnModel).__init__(config_path, models)

    def predict(self, data):
        """
        Predict data.
        :param data: any type of input data
        :return: predicted values
        """
        X = self._gen_features(data)
        return self._estimator.predict(X)

    def _save(self):
        """
        Save model to file.
        """
        # TODO create ser_dir, not 'any_dir'.
        with open('any_dir') as f:
            pickle.dump(self._estimator, f)

    @overrides
    def _load(self):
        """
        Load model from file.
        """
        # TODO decide what dir to check for loading.
        with open('any_dir') as f:
            pickle.dump(self._estimator, f)

    def _gen_features(self, dataset):
        """
        Generate necessary for training features. Use :attr:`models`to generate input feature
        vector. The function should return an input feature vector.
        :param dataset: a dataset used for training
        """
        # TODO should return target vector only for train.
        raise NotImplementedError

    def train(self, dataset):
        X = self._gen_features(dataset)
        self._estimator.fit(X)
        self._save()
