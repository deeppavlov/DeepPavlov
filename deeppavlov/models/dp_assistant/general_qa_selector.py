from logging import getLogger
from typing import Tuple, List
import pickle

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable

log = getLogger(__name__)


@register('general_qa_selector')
class GeneralQASelector(Component, Serializable):
    def __init__(self, load_path: str, debug=True, *args, **kwargs) -> None:
        super(GeneralQASelector, self).__init__(load_path=load_path, save_path=None)
        self._debug = debug
        self.count_vect = None
        self.xgb_model = None
        self.load()

    def load(self):
        self.load_path = self.load_path.expanduser()
        with open(self.load_path, 'rb') as f:
            models_dict = pickle.load(f)
            self.count_vect = models_dict['count_vectorizer']
            self.xgb_model = models_dict['xgb_model']

    def save(self):
        pass

    def __call__(self,
                 questions_raw: List[str],
                 answers_rubert: List[str],
                 answers_rubert_score: List[float],
                 answers_rubert_retr: List[str],
                 answers_rubert_retr_score: List[float],
                 answers_kbqa: List[str],
                 answers_kbqa_score: List[float]) -> List[Tuple[str, str]]:
        best_answers = []
        for q, a_odqa, c_odqa, a_odqa_retr, c_odqa_retr, a_kbqa, c_kbqa in zip(questions_raw,
                                                                               answers_rubert,
                                                                               answers_rubert_score,
                                                                               answers_rubert_retr,
                                                                               answers_rubert_retr_score,
                                                                               answers_kbqa,
                                                                               answers_kbqa_score):

            # Prepare count features
            vecs = [self.count_vect.transform([q]).todense(),
                    self.count_vect.transform([a_odqa]).todense(),
                    self.count_vect.transform([a_odqa_retr]).todense(),
                    self.count_vect.transform([a_kbqa]).todense()]
            x_counts = np.concatenate(vecs, 1)
            confidences = np.expand_dims(np.array([c_odqa, c_odqa_retr, c_kbqa]), 0)
            x_ar = self._compute_confidence_features(confidences)
            x_ar = np.concatenate([x_ar, x_counts], 1)
            model_probs = self.xgb_model.predict_proba(x_ar)
            best_model_id = np.argmax(model_probs)

            if self._is_kbqa_question(q):
                if self._debug:
                    log.debug('KBQA Template detected')
                best_model_id = 2
            answers = [a_odqa, a_odqa_retr, a_kbqa]
            if self._ved_check(q):
                best_answers.append('Первый заместитель Председателя Правления Сбербанка')
            else:
                best_answers.append(answers[best_model_id])
            if self._debug:
                log.debug({'rubert': a_odqa,
                           'rubert_retr': a_odqa_retr,
                           'kbqa': a_kbqa})
                log.debug(f'Chosen answer: {answers[best_model_id]}')
        return best_answers

    @staticmethod
    def _is_kbqa_question(q):
        q = q.lower()
        templates = ['кто такой', 'что такое', 'когда родился', 'когда умер', 'какая столица', 'столица']
        for template in templates:
            if q.startswith(template):
                return True
        return False

    @staticmethod
    def _ved_check(q):
        q = q.lower()
        return q.startswith('кто') and 'ведяхи' in q

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
