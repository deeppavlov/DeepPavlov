import unicodedata

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

from nltk import word_tokenize


@register('squad_preprocessor')
class SquadPreprocessor(Component):
    def __init__(self, *args, **kwargs):
        pass

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
            c_tokens = [token.replace("''", '"').replace("``", '"') for token in word_tokenize(c)]
            c_chars = [list(token) for token in c_tokens]
            q = SquadPreprocessor.preprocess_str(q_raw)
            q_tokens = [token.replace("''", '"').replace("``", '"') for token in word_tokenize(q)]
            q_chars = [list(token) for token in q_tokens]
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