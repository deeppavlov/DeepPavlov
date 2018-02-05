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

"""
Inherit from this model to implement a `scikit-learn <http://scikit-learn.org/stable/>`_ model.
"""
from typing import Type, Dict

from deeppavlov.core.models.inferable import Inferable
from deeppavlov.core.models.trainable import Trainable
from deeppavlov.core.common.file import load_pickle, save_pickle


class SklearnModel(Trainable, Inferable):
    def __init__(self, estimator: Type, params: Dict = None, ser_path=None, ser_dir='sklearn',
                 ser_file='estimator.pkl', train_now=False):

        super().__init__(model_path=ser_path,
                         model_dir=ser_dir,
                         model_file=ser_file,
                         train_now=train_now)

        if params is None:
            self._params = {}
        else:
            self._params = params
        self._estimator = estimator().set_params(**self._params)

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
                    "S{} estimator doesn't have predict_proba() method.".format(
                        self._estimator.__class__.__name__))

        elif prediction_type == 'log_proba':
            log_proba = getattr(self._estimator, 'log_proba', None)
            if callable(log_proba):
                return self._estimator.predict_log_proba(features, fit_params)
            else:
                raise AttributeError(
                    "{} estimator doesn't have predict_proba() method.".format(
                        self._estimator.__class__.__name__))

    def train(self, data, target, fit_params=None, *args, **kwargs):
        if fit_params is None:
            fit_params = {}
        self._estimator.fit(data, target, **fit_params)
        self.save()

    def save(self):
        """
        Save model to file.
        """
        if not self.ser_path.parent.exists():
            self.ser_path.parent.mkdir(mode=0o755)

        save_pickle(self._estimator, self.ser_path.as_posix())

        print(':: model saved to {}'.format(self.ser_path))

    def load(self):
        """
        Load model from file.
        """
        try:
            return load_pickle(self.ser_path)
        except FileNotFoundError as e:
            raise (e, "There is no model in the specified path: {}".format(self.ser_path))

    def reset(self):
        pass
