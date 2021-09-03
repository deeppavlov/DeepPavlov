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


import bisect
import pickle
import unicodedata
from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import Tuple, List, Union, Dict

import numpy as np
from nltk import word_tokenize
from tqdm import tqdm

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator

logger = getLogger(__name__)


@register('squad_preprocessor')
class SquadPreprocessor(Component):
    """ SquadPreprocessor is used to preprocess context and question in SQuAD-like datasets.

        Preprocessing includes: sanitizing unicode symbols, quotes, word tokenizing and
        building mapping from raw text to processed text.

        Params:
            context_limit: max context length in tokens
            question_limit: max question length in tokens
            char_limit: max number of characters in token
    """

    def __init__(self, context_limit: int = 450, question_limit: int = 150, char_limit: int = 16, *args, **kwargs):
        self.context_limit = context_limit
        self.question_limit = question_limit
        self.char_limit = char_limit

    def __call__(self, contexts_raw: Tuple[str, ...], questions_raw: Tuple[str, ...],
                 **kwargs) -> Tuple[
        List[str], List[List[str]], List[List[List[str]]],
        List[List[int]], List[List[int]],
        List[str], List[List[str]], List[List[List[str]]],
        List[List[Tuple[int, int]]]
    ]:
        """ Performs preprocessing of context and question
        Args:
            contexts_raw: batch of contexts to preprocess
            questions_raw: batch of questions to preprocess

        Returns:
            context: batch of processed contexts
            contexts_tokens: batch of tokenized contexts
            contexts_chars: batch of tokenized and split on chars contexts
            contexts_r2p: batch of mappings from raw context to processed context
            contexts_p2r: batch of mappings from procesesd context to raw context
            questions: batch of processed questions
            questions_tokens: batch of tokenized questions
            questions_chars: batch of tokenized and split on chars questions
            spans: batch of mapping tokens to position in context
        """
        contexts = []
        contexts_tokens = []
        contexts_chars = []
        contexts_r2p = []
        contexts_p2r = []
        questions = []
        questions_tokens = []
        questions_chars = []
        spans = []
        for c_raw, q_raw in zip(contexts_raw, questions_raw):
            c, r2p, p2r = SquadPreprocessor.preprocess_str(c_raw, return_mapping=True)
            c_tokens = [token.replace("''", '"').replace("``", '"') for token in word_tokenize(c)][:self.context_limit]
            c_chars = [list(token)[:self.char_limit] for token in c_tokens]
            q = SquadPreprocessor.preprocess_str(q_raw)
            q_tokens = [token.replace("''", '"').replace("``", '"') for token in word_tokenize(q)][:self.question_limit]
            q_chars = [list(token)[:self.char_limit] for token in q_tokens]
            contexts.append(c)
            contexts_tokens.append(c_tokens)
            contexts_chars.append(c_chars)
            contexts_r2p.append(r2p)
            contexts_p2r.append(p2r)
            questions.append(q)
            questions_tokens.append(q_tokens)
            questions_chars.append(q_chars)
            spans.append(SquadPreprocessor.convert_idx(c, c_tokens))
        return contexts, contexts_tokens, contexts_chars, contexts_r2p, contexts_p2r, \
               questions, questions_tokens, questions_chars, spans

    @staticmethod
    def preprocess_str(line: str, return_mapping: bool = False) -> Union[Tuple[str, List[int], List[int]], str]:
        """ Removes unicode and other characters from str

        Args:
            line: string to process
            return_mapping: return mapping from line to preprocessed line or not

        Returns:
            preprocessed line, raw2preprocessed mapping, preprocessed2raw mapping

        """
        if not return_mapping:
            return ''.join(c for c in line if not unicodedata.combining(c)).replace("''", '" ').replace("``", '" ')

        r2p = [len(line)] * (len(line) + 1)
        p2r = [len(line)] * (len(line) + 1)
        s = ''
        for i, c in enumerate(line):
            if unicodedata.combining(c):
                r2p[i] = -1
            else:
                s += c
                r2p[i] = len(s) - 1
                p2r[len(s) - 1] = i
        return s.replace("''", '" ').replace("``", '" '), r2p, p2r

    @staticmethod
    def convert_idx(text: str, tokens: List[str]) -> List[Tuple[int, int]]:
        current = 0
        spans = []
        for token in tokens:
            current = text.find(token, current)
            if current < 0:
                logger.error("Token {} cannot be found".format(token))
                raise Exception()
            spans.append((current, current + len(token)))
            current += len(token)
        return spans


@register('squad_ans_preprocessor')
class SquadAnsPreprocessor(Component):
    """ SquadAnsPreprocessor is responsible for answer preprocessing."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, answers_raw: Tuple[List[str], ...], answers_start: Tuple[List[int], ...],
                 r2ps: List[List[int]], spans: List[List[Tuple[int, int]]],
                 **kwargs) -> Tuple[List[List[str]], List[List[int]], List[List[int]]]:
        """ Processes answers for SQuAD dataset

        Args:
            answers_raw: list of str [batch_size x number_of_answers]
            answers_start: start position of answer (in chars) [batch_size x number_of_answers]
            r2ps: mapping from raw context to processed context
            spans: mapping tokens to position in context

        Returns:
            processed answer text, start position in tokens, end position in tokens
            [batch_size x number_of_answers]

        """
        answers = []
        start = []
        end = []
        for ans_raw, ans_st, r2p, span in zip(answers_raw, answers_start, r2ps, spans):
            start.append([])
            end.append([])
            answers.append([])
            for a_raw, a_st in zip(ans_raw, ans_st):
                ans = SquadPreprocessor.preprocess_str(a_raw)
                ans_st = r2p[a_st]
                ans_end = ans_st + len(ans)
                answer_span = []
                for idx, sp in enumerate(span):
                    if not (ans_end <= sp[0] or ans_st >= sp[1]):
                        answer_span.append(idx)
                if len(answer_span) != 0:
                    y1, y2 = answer_span[0], answer_span[-1]
                else:
                    # answer not found in context
                    y1, y2 = -1, -1
                start[-1].append(y1)
                end[-1].append(y2)
                answers[-1].append(ans)
        return answers, start, end


@register('squad_vocab_embedder')
class SquadVocabEmbedder(Estimator):
    """ SquadVocabEmbedder is used to build tokens/chars vocabulary and embedding matrix.

        It extracts tokens/chars form dataset and looks for pretrained embeddings.

        Params:
            emb_folder: path to download pretrained embeddings
            emb_url: link to pretrained embeddings
            save_path: extracted embeddings save path
            load_path: extracted embeddigns load path
            context_limit: max context length in tokens
            question_limit: max question length in tokens
            char_limit: max number of characters in token
            level: token or char
        """

    def __init__(self, emb_folder: str, emb_url: str, save_path: str, load_path: str,
                 context_limit: int = 450, question_limit: int = 150, char_limit: int = 16,
                 level: str = 'token', *args, **kwargs):
        self.emb_folder = expand_path(emb_folder)
        self.level = level
        self.emb_url = emb_url
        self.emb_file_name = Path(emb_url).name
        self.save_path = expand_path(save_path)
        self.load_path = expand_path(load_path)
        self.context_limit = context_limit
        self.question_limit = question_limit
        self.char_limit = char_limit
        self.loaded = False

        self.NULL = "<NULL>"
        self.OOV = "<OOV>"

        self.emb_folder.mkdir(parents=True, exist_ok=True)

        self.emb_dim = self.emb_mat = self.token2idx_dict = None

        if self.load_path.exists():
            self.load()

    def __call__(self, contexts: List[List[str]], questions: List[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
        """ Transforms tokens/chars to indices.

        Args:
            contexts: batch of list of tokens in context
            questions: batch of list of tokens in question

        Returns:
            transformed contexts and questions
        """
        if self.level == 'token':
            c_idxs = np.zeros([len(contexts), self.context_limit], dtype=np.int32)
            q_idxs = np.zeros([len(questions), self.question_limit], dtype=np.int32)
            for i, context in enumerate(contexts):
                for j, token in enumerate(context):
                    c_idxs[i, j] = self._get_idx(token)

            for i, question in enumerate(questions):
                for j, token in enumerate(question):
                    q_idxs[i, j] = self._get_idx(token)

        elif self.level == 'char':
            c_idxs = np.zeros([len(contexts), self.context_limit, self.char_limit], dtype=np.int32)
            q_idxs = np.zeros([len(questions), self.question_limit, self.char_limit], dtype=np.int32)
            for i, context in enumerate(contexts):
                for j, token in enumerate(context):
                    for k, char in enumerate(token):
                        c_idxs[i, j, k] = self._get_idx(char)

            for i, question in enumerate(questions):
                for j, token in enumerate(question):
                    for k, char in enumerate(token):
                        q_idxs[i, j, k] = self._get_idx(char)

        return c_idxs, q_idxs

    def fit(self, contexts: Tuple[List[str], ...], questions: Tuple[List[str]], *args, **kwargs):
        self.vocab = Counter()
        self.embedding_dict = dict()
        if not self.loaded:
            logger.info('SquadVocabEmbedder: fitting with {}s'.format(self.level))
            if self.level == 'token':
                for line in tqdm(contexts + questions):
                    for token in line:
                        self.vocab[token] += 1
            elif self.level == 'char':
                for line in tqdm(contexts + questions):
                    for token in line:
                        for c in token:
                            self.vocab[c] += 1
            else:
                raise RuntimeError("SquadVocabEmbedder::fit: Unknown level: {}".format(self.level))

            with (self.emb_folder / self.emb_file_name).open('r', encoding='utf8') as femb:
                emb_voc_size, self.emb_dim = map(int, femb.readline().split())
                for line in tqdm(femb, total=emb_voc_size):
                    line_split = line.strip().split(' ')
                    word = line_split[0]
                    vec = np.array(line_split[1:], dtype=float)
                    if len(vec) != self.emb_dim:
                        continue
                    if word in self.vocab:
                        self.embedding_dict[word] = vec

            self.token2idx_dict = {token: idx for idx, token in enumerate(self.embedding_dict.keys(), 2)}
            self.token2idx_dict[self.NULL] = 0
            self.token2idx_dict[self.OOV] = 1
            self.embedding_dict[self.NULL] = [0.] * self.emb_dim
            self.embedding_dict[self.OOV] = [0.] * self.emb_dim
            idx2emb_dict = {idx: self.embedding_dict[token]
                            for token, idx in self.token2idx_dict.items()}
            self.emb_mat = np.array([idx2emb_dict[idx] for idx in range(len(idx2emb_dict))])

    def load(self) -> None:
        logger.info('SquadVocabEmbedder: loading saved {}s vocab from {}'.format(self.level, self.load_path))
        with self.load_path.open('rb') as f:
            self.emb_dim, self.emb_mat, self.token2idx_dict = pickle.load(f)
        self.loaded = True

    def deserialize(self, data: bytes) -> None:
        self.emb_dim, self.emb_mat, self.token2idx_dict = pickle.loads(data)
        self.loaded = True

    def save(self) -> None:
        logger.info('SquadVocabEmbedder: saving {}s vocab to {}'.format(self.level, self.save_path))
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with self.save_path.open('wb') as f:
            pickle.dump((self.emb_dim, self.emb_mat, self.token2idx_dict), f, protocol=4)

    def serialize(self) -> bytes:
        return pickle.dumps((self.emb_dim, self.emb_mat, self.token2idx_dict), protocol=4)

    def _get_idx(self, el: str) -> int:
        """ Returns idx for el (token or char).

        Args:
            el: token or character

        Returns:
            idx in vocabulary
        """
        for e in (el, el.lower(), el.capitalize(), el.upper()):
            if e in self.token2idx_dict:
                return self.token2idx_dict[e]
        return 1


@register('squad_ans_postprocessor')
class SquadAnsPostprocessor(Component):
    """ SquadAnsPostprocessor class is responsible for processing SquadModel output.

        It extract answer from context using predicted by SquadModel answer positions.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, ans_start: Tuple[int, ...], ans_end: Tuple[int, ...], contexts: Tuple[str, ...],
                 p2rs: List[List[int]], spans: List[List[Tuple[int, int]]],
                 **kwargs) -> Tuple[List[str], List[int], List[int]]:
        """ Extracts answer from context using predicted answer positions.

        Args:
            ans_start: predicted start position in processed context: list of ints with len(ans_start) == batch_size
            ans_end: predicted end position in processed context
            contexts: raw contexts
            p2rs: mapping from processed context to raw
            spans: tokens positions in context

        Returns:
            postprocessed answer text, start position in raw context, end position in raw context
        """
        answers = []
        start = []
        end = []
        for a_st, a_end, c, p2r, span in zip(ans_start, ans_end, contexts, p2rs, spans):
            if a_st == -1 or a_end == -1:
                start.append(-1)
                end.append(-1)
                answers.append('')
            else:
                start.append(p2r[span[a_st][0]])
                end.append(p2r[span[a_end][1]])
                answers.append(c[start[-1]:end[-1]])
        return answers, start, end


@register('squad_bert_mapping')
class SquadBertMappingPreprocessor(Component):
    """Create mapping from BERT subtokens to their characters positions and vice versa.

        Args:
            do_lower_case: set True if lowercasing is needed

    """

    def __init__(self, do_lower_case: bool = True, *args, **kwargs):
        self.do_lower_case = do_lower_case

    def __call__(self, contexts, bert_features, *args, **kwargs):
        subtok2chars: List[Dict[int, int]] = []
        char2subtoks: List[Dict[int, int]] = []

        for batch_counter, (context, features) in enumerate(zip(contexts, bert_features)):
            subtokens: List[str]
            if self.do_lower_case:
                context = context.lower()
            if len(args) > 0:
                subtokens = args[0][batch_counter]
            else:
                subtokens = features.tokens
            context_start = subtokens.index('[SEP]') + 1
            idx = 0
            subtok2char: Dict[int, int] = {}
            char2subtok: Dict[int, int] = {}
            for i, subtok in list(enumerate(subtokens))[context_start:-1]:
                subtok = subtok[2:] if subtok.startswith('##') else subtok
                subtok_pos = context[idx:].find(subtok)
                if subtok_pos == -1:
                    # it could be UNK
                    idx += 1  # len was at least one
                else:
                    # print(k, '\t', t, p + idx)
                    idx += subtok_pos
                    subtok2char[i] = idx
                    for j in range(len(subtok)):
                        char2subtok[idx + j] = i
                    idx += len(subtok)
            subtok2chars.append(subtok2char)
            char2subtoks.append(char2subtok)
        return subtok2chars, char2subtoks


@register('squad_bert_ans_preprocessor')
class SquadBertAnsPreprocessor(Component):
    """Create answer start and end positions in subtokens.

        Args:
            do_lower_case: set True if lowercasing is needed

    """

    def __init__(self, do_lower_case: bool = True, *args, **kwargs):
        self.do_lower_case = do_lower_case

    def __call__(self, answers_raw, answers_start, char2subtoks, **kwargs):
        answers, starts, ends = [], [], []
        for answers_raw, answers_start, c2sub in zip(answers_raw, answers_start, char2subtoks):
            answers.append([])
            starts.append([])
            ends.append([])
            for ans, ans_st in zip(answers_raw, answers_start):
                if self.do_lower_case:
                    ans = ans.lower()
                try:
                    indices = {c2sub[i] for i in range(ans_st, ans_st + len(ans)) if i in c2sub}
                    st = min(indices)
                    end = max(indices)
                except ValueError:
                    # 0 - CLS token
                    st, end = 0, 0
                    ans = ''
                starts[-1] += [st]
                ends[-1] += [end]
                answers[-1] += [ans]
        return answers, starts, ends


@register('squad_bert_ans_postprocessor')
class SquadBertAnsPostprocessor(Component):
    """Extract answer and create answer start and end positions in characters from subtoken positions."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, answers_start, answers_end, contexts, bert_features, subtok2chars, *args, **kwargs):
        answers = []
        starts = []
        ends = []
        for batch_counter, (answer_st, answer_end, context, features, sub2c) in \
                enumerate(zip(answers_start, answers_end, contexts, bert_features, subtok2chars)):
            # CLS token is no_answer token
            if answer_st == 0 or answer_end == 0:
                answers += ['']
                starts += [-1]
                ends += [-1]
            else:
                st = self.get_char_position(sub2c, answer_st)
                end = self.get_char_position(sub2c, answer_end)
                if len(args) > 0:
                    subtok = args[0][batch_counter][answer_end]
                else:
                    subtok = features.tokens[answer_end]
                subtok = subtok[2:] if subtok.startswith('##') else subtok
                answer = context[st:end + len(subtok)]
                answers += [answer]
                starts += [st]
                ends += [ends]
        return answers, starts, ends

    @staticmethod
    def get_char_position(sub2c, sub_pos):
        keys = list(sub2c.keys())
        found_idx = bisect.bisect(keys, sub_pos)
        if found_idx == 0:
            return sub2c[keys[0]]

        return sub2c[keys[found_idx - 1]]
