from typing import Tuple, List
import pickle

import numpy as np
import xgboost as xgb

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


@register('qa_selector')
class QASelector(Component):
    def __init__(self, load_path: str, *args, **kwargs) -> None:
        self.count_vect = None
        self.xgb = None
        self.load(load_path)

    def load(self, file_path: str):
        with open(file_path, 'rb') as f:
            models_dict = pickle.load(f)
            self.count_vect = models_dict['count_vectorizer']
            self.xgb_model = models_dict['xgb']

    def __call__(self,
                 question_raw: List[str],
                 answer_rubert: List[str],
                 answer_rubert_score: List[float],
                 answer_rubert_retr: List[str],
                 answer_rubert_retr_score: List[float],
                 answer_kbqa: List[str],
                 answer_kbqa_score: List[float]) -> Tuple[str, str]:

        # Prepare count features
        vecs = [self.count_vect.transform(question_raw).todense(),
                self.count_vect.transform(answer_rubert).todense(),
                self.count_vect.transform(answer_rubert_retr).todense(),
                self.count_vect.transform(answer_kbqa).todense()]
        x_counts = np.concatenate(vecs, 1)
        confidences = np.stack([answer_rubert_score, answer_rubert_retr_score, answer_kbqa_score], 0)
        x_ar = self._compute_confidence_features(confidences)
        x_ar = np.concatenate([x_ar, x_counts], 1)
        d = xgb.DMatrix(x_ar)
        best_model_ids = self.xgb_model.predict(d)

        answers = [answer_rubert,
                   answer_rubert_retr,
                   answer_kbqa]
        model_names = ['rubert',
                       'rubert_retr',
                       'kbqa']
        best_answers = []
        for n, idx in enumerate(best_model_ids):
            best_answers.append([answers[idx][n], model_names[idx]])

        return best_answers

    @staticmethod
    def _compute_confidence_features(x):
        features_list = [x]

        # log
        eps = 1e-9
        l = np.log(x + eps)
        features_list.append(l)

        # square
        sq = np.square(x)
        features_list.append(sq)

        # min
        m = np.min(x, 1, keepdims=True)
        features_list.append(m)

        # max
        m = np.max(x, 1, keepdims=True)
        features_list.append(m)

        # cov
        x_0 = np.expand_dims(x, 2)
        x_1 = np.expand_dims(x, 1)
        f = np.reshape(x_0 * x_1, [-1, x_0.shape[1] ** 2])
        features_list.append(f)

        x_ar = np.concatenate(features_list, 1)
        return x_ar
