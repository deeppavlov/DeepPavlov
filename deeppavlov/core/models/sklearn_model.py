"""
Inherit from this model to implement a `scikit-learn <http://scikit-learn.org/stable/>`_ model.
"""
import pickle
from typing import Type

from overrides import overrides

from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.models.trainable import Trainable


# TODO Could inherit this from sklearn.BaseEstimator.
# TODO Could register sklearn estimator names, so developer won't need to explicitly call for estimator in code.
# Now developers can't write just "svc" in config, they have to write their own model inherited from this class
# and explicitly pass sklearn.svm.SVC class as `estimator` param to the constructor. Registering sklearn names
# would solve this issue. However, developer will have to look in the docs for registered names.
class SklearnModel(Trainable, Inferable):
    def __init__(self, models, params, estimator: Type):
        # TODO where the estimator parameters should initialize?
        self._estimator = estimator
        super(SklearnModel).__init__(models, params)

    def infer(self, data):
        """
        Load model and predict data.
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
