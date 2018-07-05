import pathlib
from collections import defaultdict
from typing import List, Dict, Generator, Tuple, Any

import numpy as np

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register


@register("dictionary_vectorizer")
class DictionaryVectorizer(Component):

    def __init__(self, load_path, min_freq=1, unk_token=None, **kwargs):
        # super().__init__(**kwargs)
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.load(load_path)

    @property
    def dim(self):
        return len(self._t2i)

    def load(self, load_path):
        if isinstance(load_path, str):
            load_path = pathlib.Path(load_path)
            if load_path.is_dir():
                load_path = [str(x) for x in load_path.iterdir() if x.is_file()]
            else:
                load_path = [str(load_path)]
        else:
            load_path = [str(x) for x in load_path]
        labels_by_words = defaultdict(set)
        for infile in load_path:
            with open(infile, "r", encoding="utf8") as fin:
                for line in fin:
                    line = line.strip()
                    if line.count("\t") != 1:
                        continue
                    word, labels = line.split("\t")
                    labels_by_words[word].update(labels.split())
        self._train(labels_by_words)
        return self

    def _train(self, labels_by_words : Dict):
        self._i2t = [self.unk_token] if self.unk_token is not None else []
        self._t2i = defaultdict(lambda: self.unk_token)
        freq = defaultdict(int)
        for word, labels in labels_by_words.items():
            for label in labels:
                freq[label] += 1
        self._i2t += [label for label, count in freq.items() if count >= self.min_freq]
        for i, label in enumerate(self._i2t):
            self._t2i[label] = i
        if self.unk_token is not None:
            self.word_tag_mapping = defaultdict(lambda: [self.unk_token])
        else:
            self.word_tag_mapping = defaultdict(list)
        for word, labels in labels_by_words.items():
            labels = {self._t2i[label] for label in labels}
            self.word_tag_mapping[word] = [x for x in labels if x is not None]
        return self

    def __call__(self, data):
        max_length = max(len(x) for x in data)
        answer = np.zeros(shape=(len(data), max_length, self.dim), dtype=int)
        for i, sent in enumerate(data):
            for j, word in enumerate(sent):
                answer[i, j][self.word_tag_mapping[word]] = 1
        return answer
