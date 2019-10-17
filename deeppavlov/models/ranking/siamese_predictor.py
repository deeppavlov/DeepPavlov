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

from logging import getLogger
from typing import List, Iterable, Callable, Union

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.core.models.component import Component
from deeppavlov.models.ranking.keras_siamese_model import SiameseModel

log = getLogger(__name__)


@register('siamese_predictor')
class SiamesePredictor(Component):
    """The class for ranking or paraphrase identification using the trained siamese network  in the ``interact`` mode.

    Args:
        batch_size: A size of a batch.
        num_context_turns: A number of ``context`` turns in data samples.
        ranking: Whether to perform ranking.
            If it is set to ``False`` paraphrase identification will be performed.
        attention: Whether any attention mechanism is used in the siamese network.
            If ``False`` then calculated in advance vectors of ``responses``
            will be used to obtain similarity score for the input ``context``;
            Otherwise the whole siamese architecture will be used
            to obtain similarity score for the input ``context`` and each particular ``response``.
            The parameter will be used if the ``ranking`` is set to ``True``.
        responses: A instance of :class:`~deeppavlov.core.data.simple_vocab.SimpleVocabulary`
            with all possible ``responses`` to perform ranking.
            Will be used if the ``ranking`` is set to ``True``.
        preproc_func: A ``__call__`` function of the
            :class:`~deeppavlov.models.preprocessors.siamese_preprocessor.SiamesePreprocessor`.
        interact_pred_num: The number of the most relevant ``responses`` which will be returned.
            Will be used if the ``ranking`` is set to ``True``.
        **kwargs: Other parameters.
    """

    def __init__(self,
                 model: SiameseModel,
                 batch_size: int,
                 num_context_turns: int = 1,
                 ranking: bool = True,
                 attention: bool = False,
                 responses: SimpleVocabulary = None,
                 preproc_func: Callable = None,
                 interact_pred_num: int = 3,
                 *args, **kwargs) -> None:

        super().__init__()

        self.batch_size = batch_size
        self.num_context_turns = num_context_turns
        self.ranking = ranking
        self.attention = attention
        self.preproc_responses = []
        self.response_embeddings = None
        self.preproc_func = preproc_func
        self.interact_pred_num = interact_pred_num
        self.model = model
        if self.ranking:
            self.responses = {el[1]: el[0] for el in responses.items()}
            self._build_preproc_responses()
            if not self.attention:
                self._build_response_embeddings()

    def __call__(self, batch: Iterable[List[np.ndarray]]) -> List[Union[List[str], str]]:
        context = next(batch)
        try:
            next(batch)
            log.error("It is not intended to use the `%s` with the batch size greater then 1." % self.__class__)
        except StopIteration:
            pass

        if self.ranking:
            if len(context) == self.num_context_turns:
                scores = []
                if self.attention:
                    for i in range(len(self.preproc_responses) // self.batch_size + 1):
                        responses = self.preproc_responses[i * self.batch_size: (i + 1) * self.batch_size]
                        b = [context + el for el in responses]
                        b = self.model._make_batch(b)
                        sc = self.model._predict_on_batch(b)
                        scores += list(sc)
                else:
                    b = self.model._make_batch([context])
                    context_emb = self.model._predict_context_on_batch(b)
                    context_emb = np.squeeze(context_emb, axis=0)
                    scores = context_emb @ self.response_embeddings.T
                ids = np.flip(np.argsort(scores), -1)
                return [[self.responses[el] for el in ids[:self.interact_pred_num]]]
            else:
                return ["Please, provide contexts separated by '&' in the number equal to that used while training."]

        else:
            if len(context) == 2:
                b = self.model._make_batch([context])
                sc = self.model._predict_on_batch(b)[0]
                if sc > 0.5:
                    return ["This is a paraphrase."]
                else:
                    return ["This is not a paraphrase."]
            else:
                return ["Please, provide two sentences separated by '&'."]

    def reset(self) -> None:
        pass

    def process_event(self) -> None:
        pass

    def _build_response_embeddings(self) -> None:
        resp_vecs = []
        for i in range(len(self.preproc_responses) // self.batch_size + 1):
            resp_preproc = self.preproc_responses[i * self.batch_size: (i + 1) * self.batch_size]
            resp_preproc = self.model._make_batch(resp_preproc)
            resp_preproc = resp_preproc
            resp_vecs.append(self.model._predict_response_on_batch(resp_preproc))
        self.response_embeddings = np.vstack(resp_vecs)

    def _build_preproc_responses(self) -> None:
        responses = list(self.responses.values())
        for i in range(len(responses) // self.batch_size + 1):
            el = self.preproc_func(responses[i * self.batch_size: (i + 1) * self.batch_size])
            self.preproc_responses += list(el)

    def rebuild_responses(self, candidates) -> None:
        self.attention = True
        self.interact_pred_num = 1
        self.preproc_responses = list()
        self.responses = {idx: sentence for idx, sentence in enumerate(candidates)}
        self._build_preproc_responses()
