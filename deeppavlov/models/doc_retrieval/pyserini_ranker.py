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

import json
import time
from logging import getLogger
from typing import List, Any, Tuple

from pyserini.search import SimpleSearcher

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component

logger = getLogger(__name__)


@register("pyserini_ranker")
class PyseriniRanker(Component):
    def __init__(self, index_folder: str, n_threads: int = 1, top_n: int = 5,
                       text_column_name: str = "contents", return_scores: bool = False, *args, **kwargs):
        self.searcher = SimpleSearcher(index_folder)
        self.n_threads = n_threads
        self.top_n = top_n
        self.text_column_name = text_column_name
        self.return_scores = return_scores
    def __call__(self, questions: List[str]) -> Tuple[List[Any], List[float]]:
        docs_batch = []
        scores_batch = []
        if self.n_threads == 1 or len(questions) == 1:
            for question in questions:
                docs = []
                scores = []
                res = self.searcher(question, self.top_n)
                for elem in res:
                    doc = json.loads(elem.raw)
                    score = elem.score
                    if doc and isinstance(doc, dict):
                        docs.append(doc.get("contents", ""))
                        scores.append(score)
                docs_batch.append(docs)
                scores_batch.append(scores)
        else:
            n_batches = len(questions) // self.n_threads + int(len(questions)%self.n_threads > 0)
            for i in range(n_batches):
                questions_cur = questions[i*self.n_threads:(i+1)*self.n_threads]
                qids_cur = list(range(len(questions_cur)))
                res_batch = self.searcher.batch_search(questions_cur, qids_cur, self.top_n, self.n_threads)
                for qid in qids_cur:
                    docs = []
                    scores = []
                    res = res_batch.get(qid)
                    for elem in res:
                        doc = json.loads(elem.raw)
                        score = elem.score
                        if doc and isinstance(doc, dict):
                            docs.append(doc.get("contents", ""))
                            scores.append(score)
                    docs_batch.append(docs)
                    scores_batch.append(scores)
        
        if self.return_scores:
            return docs_batch, scores_batch
        else:
            return docs_batch
