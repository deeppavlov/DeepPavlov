import unicodedata
from collections import Counter
import pickle

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.data.utils import download
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.estimator import Estimator

from nltk import word_tokenize
from tqdm import tqdm
import numpy as np

from deeppavlov.core.common.log import get_logger

logger = get_logger(__name__)


@register('squad_preprocessor')
class SquadPreprocessor(Component):
    def __init__(self, context_limit, question_limit, char_limit, *args, **kwargs):
        self.context_limit = context_limit
        self.question_limit = question_limit
        self.char_limit = char_limit

    def __call__(self, contexts_raw, questions_raw, **kwargs):
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
    def preprocess_str(line, return_mapping=False):
        """Removes unicode and other characters from str

        :param line:
        :param return_mapping: return mapping from line to preprocessed line
        :return: preprocessed line, raw2preprocessed, preprocessed2raw
        """
        line = line.replace("''", '" ').replace("``", '" ')
        if not return_mapping:
            return ''.join(c for c in line if not unicodedata.combining(c))

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
        return s, r2p, p2r

    @staticmethod
    def convert_idx(text, tokens):
        current = 0
        spans = []
        for token in tokens:
            current = text.find(token, current)
            if current < 0:
                print("Token {} cannot be found".format(token))
                raise Exception()
            spans.append((current, current + len(token)))
            current += len(token)
        return spans


@register('squad_ans_preprocessor')
class SquadAnsPreprocessor(Component):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, ans_raw, ans_start, r2ps, spans, **kwargs):
        """Processes answers for SQuAD dataset

        :param ans_raw: answer text
        :param ans_start: start position of answer (in chars)
        :param r2ps: mapping from raw context to processed
        :param spans:
        :return: processed answer text, start position in tokens, end position in tokens
        """
        answers = []
        answers_start = []
        answers_end = []
        for a_raw, a_st, r2p, span in zip(ans_raw, ans_start, r2ps, spans):
            ans = SquadPreprocessor.preprocess_str(a_raw)
            ans_st = r2p[a_st]
            ans_end = ans_st + len(ans)
            answer_span = []
            for idx, sp in enumerate(span):
                if not (ans_end <= sp[0] or ans_st >= sp[1]):
                    answer_span.append(idx)
            y1, y2 = answer_span[0], answer_span[-1]
            answers_start.append(y1)
            answers_end.append(y2)
            answers.append(ans)
        return answers, answers_start, answers_end


@register('squad_vocab_embedder')
class SquadVocabEmbedder(Estimator):
    def __init__(self, emb_folder, emb_url, save_path, load_path,
                 context_limit, question_limit, char_limit, level='token', *args, **kwargs):
        self.emb_folder = expand_path(emb_folder)
        self.level = level
        self.emb_url = emb_url
        self.emb_file_name = '{}_emb.vec'.format(self.level)
        self.save_path = expand_path(save_path)
        self.load_path = expand_path(load_path)
        self.context_limit = context_limit
        self.question_limit = question_limit
        self.char_limit = char_limit
        self.loaded = False

        self.NULL = "<NULL>"
        self.OOV = "<OOV>"

        if not self.emb_folder.exists():
            self.emb_folder.mkdir(parents=True)

        if not (self.emb_folder / self.emb_file_name).exists():
            download(self.emb_folder / self.emb_file_name, self.emb_url)

        if self.load_path.exists():
            self.load()

    def __call__(self, contexts, questions):
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

    def fit(self, contexts, questions, *args, **kwargs):
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

            with (self.emb_folder / self.emb_file_name).open('r') as femb:
                emb_voc_size, self.emb_dim = map(int, femb.readline().split())
                for line in tqdm(femb, total=emb_voc_size):
                    line_split = line.strip().split(' ')
                    word = line_split[0]
                    vec = np.array(line_split[1:], dtype=float)
                    if len(vec) != self.emb_dim:
                        continue
                    if word in self.vocab:
                        self.embedding_dict[word] = vec

            self.token2idx_dict = {token: idx for idx,
                                             token in enumerate(self.embedding_dict.keys(), 2)}
            self.token2idx_dict[self.NULL] = 0
            self.token2idx_dict[self.OOV] = 1
            self.embedding_dict[self.NULL] = [0. for _ in range(self.emb_dim)]
            self.embedding_dict[self.OOV] = [0. for _ in range(self.emb_dim)]
            idx2emb_dict = {idx: self.embedding_dict[token]
                            for token, idx in self.token2idx_dict.items()}
            self.emb_mat = np.array([idx2emb_dict[idx] for idx in range(len(idx2emb_dict))])

    def load(self, *args, **kwargs):
        logger.info('SquadVocabEmbedder: loading saved {}s vocab from {}'.format(self.level, self.load_path))
        self.emb_dim, self.emb_mat, self.token2idx_dict = pickle.load(self.load_path.open('rb'))
        self.loaded = True

    def save(self, *args, **kwargs):
        logger.info('SquadVocabEmbedder: saving {}s vocab to {}'.format(self.level, self.save_path))
        pickle.dump((self.emb_dim, self.emb_mat, self.token2idx_dict), self.save_path.open('wb'))

    def _get_idx(self, el):
        """Returns idx for el (token or char).
        If el is not in vocabulary then looks for el.lower() and etc

        :param el: token or character
        :return: idx in vocabulary
        """
        for e in (el, el.lower(), el.capitalize(), el.upper()):
            if e in self.token2idx_dict:
                return self.token2idx_dict[e]
        return 1
