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

import time
from logging import getLogger
from operator import itemgetter
from typing import List, Union, Tuple, Optional

from deeppavlov.core.common.chainer import Chainer
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component
from deeppavlov.models.doc_retrieval.utils import find_answer_sentence
from deeppavlov.models.preprocessors.odqa_preprocessors import StringMultiplier

logger = getLogger(__name__)


@register("logit_ranker")
class LogitRanker(Component):
    """Select best answer using squad model logits. Make several batches for a single batch, send each batch
     to the squad model separately and get a single best answer for each batch.

     Args:
        squad_model: a loaded squad model
        batch_size: batch size to use with squad model
        sort_noans: whether to downgrade noans tokens in the most possible answers
        top_n: number of answers to return

     Attributes:
        squad_model: a loaded squad model
        batch_size: batch size to use with squad model
        top_n: number of answers to return

    """

    def __init__(self, squad_model: Union[Chainer, Component], batch_size: int = 50,
                 sort_noans: bool = False, top_n: int = 1, return_answer_sentence: bool = False,
                 use_topical_chat: bool = False, string_multiplier: StringMultiplier = None, **kwargs):
        self.squad_model = squad_model
        self.batch_size = batch_size
        self.sort_noans = sort_noans
        self.top_n = top_n
        self.return_answer_sentence = return_answer_sentence
        self.use_topical_chat = use_topical_chat
        self.string_multiplier = string_multiplier

    def __call__(self, questions_batch: List[str],
                       contexts_batch_wiki: List[List[str]],
                       contexts_batch_tch: List[List[str]],
                       doc_ids_batch: Optional[List[List[str]]] = None) -> \
            Union[
                Tuple[List[str], List[float], List[int], List[str]],
                Tuple[List[List[str]], List[List[float]], List[List[int]], List[List[str]]],
                Tuple[List[str], List[float], List[int]],
                Tuple[List[List[str]], List[List[float]], List[List[int]]]
            ]:

        """
        Sort obtained results from squad reader by logits and get the answer with a maximum logit.

        Args:
            contexts_batch: a batch of contexts which should be treated as a single batch in the outer JSON config
            questions_batch: a batch of questions which should be treated as a single batch in the outer JSON config
            doc_ids_batch (optional): names of the documents from which the contexts_batch was derived
        Returns:
             a batch of best answers, their scores, places in contexts
             and doc_ids for this answers if doc_ids_batch were passed
        """
        if doc_ids_batch is None:
            logger.warning("you didn't pass tfidf_doc_ids as input in logit_ranker config so "
                           "batch_best_answers_doc_ids can't be compute")
        questions_batch_wiki = self.string_multiplier(questions_batch, contexts_batch_wiki)

        tm_st = time.time()
        batch_best_answers_wiki, batch_best_answers_score_wiki, batch_best_answers_place_wiki, \
                   batch_best_answers_sentences_wiki, batch_best_answers_contexts_wiki = self.find_answer(
                                                                        contexts_batch_wiki, questions_batch_wiki)
        tm_end = time.time()
        logger.debug(f"time of wikipedia squad {tm_end-tm_st}")
        if self.use_topical_chat:
            tm_st = time.time()
            questions_batch_tch = self.string_multiplier(questions_batch, contexts_batch_tch)
            batch_best_answers_tch, batch_best_answers_score_tch, batch_best_answers_place_tch, \
                       batch_best_answers_sentences_tch, batch_best_answers_contexts_tch = self.find_answer(
                                                                            contexts_batch_tch, questions_batch_tch)
            tm_end = time.time()
            logger.debug(f"time of topical chat squad {tm_end-tm_st}")
            if self.return_answer_sentence:
                return batch_best_answers_wiki, batch_best_answers_score_wiki, batch_best_answers_place_wiki, \
                       batch_best_answers_sentences_wiki, batch_best_answers_sentences_tch, \
                       batch_best_answers_contexts_wiki
        if self.return_answer_sentence:
                return batch_best_answers_wiki, batch_best_answers_score_wiki, batch_best_answers_place_wiki, \
                       batch_best_answers_sentences_wiki, batch_best_answers_contexts_wiki
        return batch_best_answers_wiki, batch_best_answers_score_wiki, batch_best_answers_place_wiki
        
    def find_answer(self, contexts_batch, questions_batch):
        batch_best_answers = []
        batch_best_answers_score = []
        batch_best_answers_place = []
        batch_best_answers_doc_ids = []
        batch_best_answers_sentences = []
        batch_best_answers_contexts = []
        for quest_ind, [contexts, questions] in enumerate(zip(contexts_batch, questions_batch)):
            results = []
            for i in range(0, len(contexts), self.batch_size):
                c_batch = contexts[i: i + self.batch_size]
                q_batch = questions[i: i + self.batch_size]
                batch_predict = list(zip(*self.squad_model(c_batch, q_batch), c_batch))
                results += batch_predict
            if self.sort_noans:
                results_sort = sorted(results, key=lambda x: (x[0] != '', x[2]), reverse=True)
            else:
                results_sort = sorted(results, key=itemgetter(2), reverse=True)
            best_answers = [x[0] for x in results_sort[:self.top_n]]
            best_answers_place = [x[1] for x in results_sort[:self.top_n]]
            best_answers_score = [x[2] for x in results_sort[:self.top_n]]
            best_answers_contexts = [x[3] for x in results_sort[:self.top_n]]
            batch_best_answers.append(best_answers)
            batch_best_answers_place.append(best_answers_place)
            batch_best_answers_score.append(best_answers_score)
            best_answers_sentences = []
            for answer, place, context in zip(best_answers, best_answers_place, best_answers_contexts):
                sentence = find_answer_sentence(place, answer, context)
                best_answers_sentences.append(sentence)
            batch_best_answers_sentences.append(best_answers_sentences)
            batch_best_answers_contexts.append(best_answers_contexts)
        
        if self.top_n == 1:
            if batch_best_answers and batch_best_answers[0]:
                batch_best_answers = [x[0] for x in batch_best_answers]
                batch_best_answers_place = [x[0] for x in batch_best_answers_place]
                batch_best_answers_score = [x[0] for x in batch_best_answers_score]
                batch_best_answers_doc_ids = [x[0] for x in batch_best_answers_doc_ids]
                batch_best_answers_sentences = [x[0] for x in batch_best_answers_sentences]
                batch_best_answers_contexts = [x[0] for x in batch_best_answers_contexts]
            else:
                batch_best_answers = ["" for _ in batch_best_answers]
                batch_best_answers_place = [0 for _ in batch_best_answers_place]
                batch_best_answers_score = [0.0 for _ in batch_best_answers_score]
                batch_best_answers_doc_ids = ["" for _ in batch_best_answers_doc_ids]
                batch_best_answers_sentences = ["" for _ in batch_best_answers_sentences]
                batch_best_answers_contexts = ["" for _ in batch_best_answers_contexts]
        
        return batch_best_answers, batch_best_answers_score, batch_best_answers_place, \
                   batch_best_answers_sentences, batch_best_answers_contexts
