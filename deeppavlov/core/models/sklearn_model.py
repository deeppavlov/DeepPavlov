"""
Inherit from this model to implement a `scikit-learn <http://scikit-learn.org/stable/>`_ model.
"""
from pathlib import Path

from typing import Type, Dict

from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.common.file import load_pickle, save_pickle
from deeppavlov.core.common import paths


class SklearnModel(Trainable, Inferable):
    def __init__(self, estimator: Type, params: Dict=None, model_dir_path='sklearn',
                 model_fpath='estimator.pkl'):
        if params is None:
            self._params = {}
        else:
            self._params = params
        self._estimator = estimator().set_params(**self._params)
        self.model_dir_path = model_dir_path
        self.model_fpath = model_fpath
        self.model_path = Path(paths.USR_PATH).joinpath(model_dir_path, model_fpath)



    def infer(self, features, fit_params=None, prediction_type='label'):
        """
        :param prediction_type: Specify type of prediction type. Sklearn estimators can predict labels,
        probas and log probas. Choose value from ['label', 'proba', 'log_proba'].
        """
        if fit_params is None:
            fit_params = {}

        fit_transform = getattr(self._estimator, 'fit_transform', None)
        if callable(fit_transform):
            return self._estimator.fit_transform(features, fit_params)

        if prediction_type == 'label':
            return self._estimator.predict(features, fit_params)

        elif prediction_type == 'proba':
            predict_proba = getattr(self._estimator, 'predict_proba', None)
            if callable(predict_proba):
                return self._estimator.predict_proba(features, fit_params)
            else:
                raise AttributeError(
                    "Scikit-learn estimator {} doesn't have predict_proba() method.".format(
                        self._estimator.__class__.__name__))

        elif prediction_type == 'log_proba':
            log_proba = getattr(self._estimator, 'log_proba', None)
            if callable(log_proba):
                return self._estimator.predict_log_proba(features, fit_params)
            else:
                raise AttributeError(
                    "Scikit-learn estimator {} doesn't have predict_proba() method.".format(
                        self._estimator.__class__.__name__))

    def train(self, data, fit_params=None, *args, **kwargs):
        features = []
        target = []
        for item in data:
            target.append(item[-1])
            features.append(item[:-1])

        if fit_params is None:
            fit_params = {}
        self._estimator.fit(X=features, y=target, **fit_params)
        self.save()

    def save(self):
        """
        Save model to file.
        """
        if not self.model_path.parent.exists():
            Path.mkdir(self.model_path.parent)

        save_pickle(self._estimator, self.model_path.as_posix())

        print(':: model saved to {}'.format(self.model_path))

    def load(self):
        """
        Load model from file.
        """
        try:
            return load_pickle(self.model_path)
        except FileNotFoundError as e:
            raise(e, "There is no model in the specified path: {}".format(self.model_path))

    def reset(self):
        pass

