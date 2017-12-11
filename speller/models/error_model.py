import csv
from collections import defaultdict, Counter

import itertools

import os
from heapq import heappop, heappushpop, heappush

from math import log, exp

from deeppavlov.common.registry import register_model
from deeppavlov.models.inferable import Inferable
from deeppavlov.models.trainable import Trainable
from speller.models.static_dictionary import StaticDictionary


@register_model('spelling_error_model')
class ErrorModel(Inferable, Trainable):
    def __init__(self, models_path, dictionary: StaticDictionary, model_name='error_model', *args, **kwargs):
        self.file_name = os.path.join(models_path, model_name + '.tsv')
        self.costs = defaultdict(itertools.repeat(float('-inf')).__next__)
        self.dictionary = dictionary
        self.costs[('', '')] = log(1)
        self.costs[('⟬', '⟬')] = log(1)
        self.costs[('⟭', '⟭')] = log(1)
        for c in self.dictionary.alphabet:
            self.costs[(c, c)] = log(1)
        if os.path.isfile(self.file_name):
            self.load(self.file_name)

    def find_candidates(self, word, k=1, prop_threshold=1e-6):
        threshold = log(prop_threshold)
        d = {}
        prefixes_heap = [(0, {''})]
        candidates = [(float('-inf'), '') for _ in range(k)]
        word = '⟬{}⟭'.format(word.lower().replace('ё', 'е'))
        word_len = len(word) + 1
        while prefixes_heap and -prefixes_heap[0][0] > candidates[0][0]:
            _, prefixes = heappop(prefixes_heap)
            for prefix in prefixes:
                res = []
                for i in range(word_len):
                    c = word[i - 1:i]
                    res.append(max(
                        (res[-1] + self.costs[('', c)]) if i else float('-inf'),
                        d[prefix[:-1]][i] + self.costs[(prefix[-1], '')] if prefix else float('-inf'),
                        (d[prefix[:-1]][i - 1] + (self.costs[(prefix[-1], c)]))
                        if prefix and i else float('-inf')
                    ) if i or prefix else 0)
                d[prefix] = res
                if prefix in self.dictionary.words_set:
                    heappushpop(candidates, (res[-1], prefix))
                potential = max(res)
                if potential > threshold:
                    heappush(prefixes_heap, (-potential, self.dictionary.words_trie[prefix]))
        return [(w.strip('⟬⟭'), exp(score)) for score, w in sorted(candidates, reverse=True)]

    def infer(self, instance: str, *args, **kwargs):
        corrected = []
        for incorrect in instance.split():
            if any([c not in self.dictionary.alphabet for c in incorrect]):
                corrected.append(incorrect)
            else:
                res, score = self.find_candidates(incorrect, k=1, prop_threshold=1e-6)[0]
                corrected.append(res if score > 1e-6 else incorrect)
        return ' '.join(corrected)

    def reset(self):
        pass

    def interact(self):

        # get input from user
        context = input(':: ')

        # check if user wants to begin new session
        if context == 'clear' or context == 'reset' or context == 'restart':
            self.reset()
            print('')

        # check for exit command
        elif context == 'exit' or context == 'stop' or context == 'quit' or context == 'q':
            return

        else:
            # ENTER press : silence
            if not context:
                context = '<SILENCE>'

            # forward
            pred = self.infer(context)
            print('>>', pred)

    @staticmethod
    def _distance_edits(seq1, seq2):
        l1, l2 = len(seq1), len(seq2)
        d = [[(i, ()) for i in range(l2 + 1)]]
        d += [[(i, ())] + [(0, ())]*l2 for i in range(1, l1 + 1)]

        for i in range(1, l1 + 1):
            for j in range(1, l2 + 1):
                edits = [
                    (d[i-1][j][0] + 1, d[i-1][j][1] + ((seq1[i-1], ''),)),
                    (d[i][j-1][0] + 1, d[i][j-1][1] + (('', seq2[j-1]),)),
                    (d[i-1][j-1][0] + (seq1[i-1] != seq2[j-1]), d[i-1][j-1][1] + ((seq1[i-1], seq2[j-1]),))
                ]
                if i > 1 and j > 1 and seq1[i-1] == seq2[j-2] and seq1[i-2] == seq2[j-1]:
                    edits.append((d[i-2][j-2][0] + (seq1[i-1] != seq2[j-1]), d[i-2][j-2][1] + ((seq1[i-2:i], seq2[j-2:j]),)))
                d[i][j] = min(edits, key=lambda x: x[0])

        return d[-1][-1]

    def train(self, data, *args, **kwargs):
        changes = []
        entries = []
        data = list(data)
        n = len(data)
        window = 4
        for i, (error, correct) in enumerate(data):
            correct = '⟬{}⟭'.format(correct)
            error = '⟬{}⟭'.format(error)
            d, ops = self._distance_edits(correct, error)
            if d <= 2:
                w_ops = set()
                for pos in range(len(ops)):
                    left, right = list(zip(*ops))
                    for l in range(pos, max(0, pos - window) - 1, -1):
                        for r in range(pos + 1, min(len(ops), l + 2 + window)):
                            w_ops.add(((''.join(left[l:r]), ''.join(right[l:r])), l, r))
                ops = [x[0] for x in w_ops]

                entries += [op[0] for op in ops]
                changes += [op for op in ops]
            if i % 1500 == 0:
                print('{} out of {}'.format(i+1, n))

        e_count = Counter(entries)
        c_count = Counter(changes)
        incorrect_prior = 1
        correct_prior = 19
        for (w, s), c in c_count.items():
            c = c + (incorrect_prior if w != s else correct_prior)
            e = e_count[w] + incorrect_prior + correct_prior
            p = c / e
            self.costs[(w, s)] = log(p)

    def save(self, file_name=None):
        if not file_name:
            file_name = self.file_name
        os.makedirs(os.path.dirname(file_name), 0o755, exist_ok=True)
        with open(file_name, 'w', newline='') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t')
            for (w, s), log_p in self.costs.items():
                writer.writerow([w, s, exp(log_p)])

    def load(self, file_name=None):
        if not file_name:
            file_name = self.file_name
        with open(file_name, 'r', newline='') as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')
            for w, s, p in reader:
                self.costs[(w, s)] = log(float(p))
