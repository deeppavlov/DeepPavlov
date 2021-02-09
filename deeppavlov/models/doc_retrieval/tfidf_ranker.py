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

import re
import time
from logging import getLogger
from typing import List, Any, Tuple

import numpy as np

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.models.vectorizers.hashing_tfidf_vectorizer import HashingTfIdfVectorizer

logger = getLogger(__name__)


@register("tfidf_ranker")
class TfidfRanker(Component):
    """Rank documents according to input strings.

    Args:
        vectorizer: a vectorizer class
        top_n: a number of doc ids to return
        active: whether to return a number specified by :attr:`top_n` (``True``) or all ids
         (``False``)

    Attributes:
        top_n: a number of doc ids to return
        vectorizer: an instance of vectorizer class
        active: whether to return a number specified by :attr:`top_n` or all ids
        index2doc: inverted :attr:`doc_index`
        iterator: a dataset iterator used for generating batches while fitting the vectorizer

    """

    def __init__(self, vectorizer: HashingTfIdfVectorizer, top_n=5, active: bool = True, **kwargs):

        self.top_n = top_n
        self.vectorizer = vectorizer
        self.active = active

    def __call__(self, questions: List[str]) -> Tuple[List[Any], List[float]]:
        """Rank documents and return top n document titles with scores.

        Args:
            questions: list of queries used in ranking

        Returns:
            a tuple of selected doc ids and their scores
        """

        batch_doc_ids, batch_docs_scores = [], []

        tm_st = time.time()
        q_tfidfs = self.vectorizer(questions)

        for q_tfidf in q_tfidfs:
            scores = q_tfidf * self.vectorizer.tfidf_matrix
            scores = np.squeeze(
                scores.toarray() + 0.0001)  # add a small value to eliminate zero scores

            if self.active:
                thresh = self.top_n
            else:
                thresh = len(self.vectorizer.doc_index)

            if thresh >= len(scores):
                o = np.argpartition(-scores, len(scores) - 1)[0:thresh]
            else:
                o = np.argpartition(-scores, thresh)[0:thresh]
            o_sort = o[np.argsort(-scores[o])]

            doc_scores = scores[o_sort]
            doc_ids = [self.vectorizer.index2doc[i] for i in o_sort]
            batch_doc_ids.append(doc_ids)
            batch_docs_scores.append(doc_scores)
        tm_end = time.time()
        logger.debug(f"tfidf ranker time {tm_end-tm_st}")
        logger.debug(f"doc ids {batch_doc_ids}")

        return batch_doc_ids, batch_docs_scores
        
        
@register("par_tfidf_ranker")
class ParTfidfRanker(Component):
    def __init__(self, tokenizer: Component,
                       np_facts_filename: str,
                       facts_map_filename: str,
                       unigrams_filename: str,
                       top_n: int = 10, log: bool = False, **kwargs):
        self.tokenizer = tokenizer
        self.np_facts = read_json(expand_path(np_facts_filename))
        self.facts_map = read_json(expand_path(facts_map_filename))
        self.top_n = top_n
        self.log = log
        np_remove_list = ["'s", 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
                  "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
                  'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',
                  'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
                  'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                  'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                  'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
                  'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                  'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
                  'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                  'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should',
                  "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
                  "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                  "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
                  'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
                  'wouldn', "wouldn't", "my name", "your name", "wow", "yeah", "yes", "ya", "cool", "okay", "more",
                  "some more", " a lot", "a bit", "another one", "something else", "something", "anything",
                  "someone", "anyone", "play", "mean", "a lot", "a little", "a little bit"]

        np_ignore_list = ["boring", "radio", "type", "call", "fun", "fall", "name", "names", "lgbtq families", "day", "murder",
                  "amazon", "take", "interest", "days", "year", "years", "sort", "fan", "going", "death", "part", "end",
                  "watching", "thought", "thoughts", "man", "men", "listening", "big fan", "fans", "rapping", "reading",
                  "going", "thing", "hanging", "best thing", "wife", "things", "nothing", "everything"]
                  
        with open(expand_path(unigrams_filename), "r") as f:
            freq_unigrams = f.read().splitlines()[:1000]
                  
        self.np_ignore_expr = re.compile("(" + "|".join([r'\b%s\b' % word for word in np_ignore_list + freq_unigrams]) + ")",
                                    re.IGNORECASE)
        self.np_remove_expr = re.compile("(" + "|".join([r'\b%s\b' % word for word in np_remove_list]) + ")", re.IGNORECASE)
        self.rm_spaces_expr = re.compile(r'\s\s+')

    def __call__(self, questions_batch: List[str],
                       paragraphs_batch: List[List[str]],
                       nounphrases_batch: List[List[str]]) -> Tuple[List[Any], List[float]]:
        batch_top_paragraphs = []
        batch_top_facts = []
        sources = []
        tm_st = time.time()
        for question, paragraphs, nounphrases_list in zip(questions_batch, paragraphs_batch, nounphrases_batch):
            facts_list = self.find_facts(nounphrases_list)
            batch_top_facts.append(facts_list)
            paragraphs = self.rank_paragraphs(question, paragraphs)
            batch_top_paragraphs.append(paragraphs)
        paragraph_total_length = sum([len(chunk) for chunk in batch_top_paragraphs[0]])
        tm_end = time.time()
        logger.debug(f"paragraph ranking time {tm_end-tm_st}, length {paragraph_total_length}")

        return batch_top_paragraphs, batch_top_facts
        
    def rank_paragraphs(self, question: str, paragraphs: List[str]) -> List[str]:
        ngrams = list(self.tokenizer([question]))[0]
        idf_scores = []
        
        for paragraph in paragraphs:
            ngrams_counts = [len(re.findall(f"{ngram}\W", paragraph, re.IGNORECASE)) for ngram in ngrams]
            non_zero_counts = [(ngram, ngram_count)
                                for ngram, ngram_count in zip(ngrams, ngrams_counts) if ngram_count > 0]
            if non_zero_counts:
                par_ngrams, ngrams_counts = zip(*non_zero_counts)
                idf_scores.append(sum(ngrams_counts))
            else:
                idf_scores.append(0.0)
        
        indices = np.argsort(idf_scores)[::-1][:self.top_n]
        top_paragraphs = [paragraphs[ind] for ind in indices]
        if self.log:
            out = open("paragraph_log.txt", 'a')
            for paragraph in top_paragraphs:
                out.write(str(paragraph)+'\n')
                out.write("_"*30+'\n')
            out.write("="*50+'\n')
            out.close()
        return top_paragraphs
            
    def find_facts(self, nounphrases_list: List[str]):
        for i in range(len(nounphrases_list)):
            nph = re.sub(self.np_remove_expr, "", nounphrases_list[i])
            nph = re.sub(self.rm_spaces_expr, " ", nph)
            if re.search(self.np_ignore_expr, nph):
                nounphrases_list[i] = ""
            else:
                nounphrases_list[i] = nph.strip()
        
        nounphrases_list = [nph for nph in nounphrases_list if len(nph) > 0]
        facts_list = []
        for i, nphrase in enumerate(nounphrases_list):
            for fact_id in self.np_facts.get(nphrase, []):
                facts_list.append(self.facts_map[str(fact_id)])
                
        return facts_list
