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

import pathlib
from abc import abstractmethod
from collections import defaultdict
from typing import List, Dict, AnyStr, Union

import numpy as np
from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.models.morpho_tagger.common_tagger import make_pos_and_tag


class WordIndexVectorizer(Serializable, Component):
    """
    A basic class for custom word-level vectorizers
    """

    def __init__(self, save_path: str, load_path: Union[str, List[str]], **kwargs) -> None:
        Serializable.__init__(self, save_path, load_path, **kwargs)

    @property
    @abstractmethod
    def dim(self):
        raise NotImplementedError("You should implement dim property in your WordIndexVectorizer subclass.")

    def _get_word_indexes(self, word: AnyStr) -> List:
        """
        Transforms a word to corresponding vector of indexes
        """
        raise NotImplementedError("You should implement get_word_indexes function "
                                  "in your WordIndexVectorizer subclass.")

    def __call__(self, data: List) -> np.ndarray:
        """
        Transforms words to one-hot encoding according to the dictionary.

        Args:
            data: the batch of words

        Returns:
            a 3D array. answer[i][j][k] = 1 iff data[i][j] is the k-th word in the dictionary.
        """
        # if isinstance(data[0], str):
        #     data = [[x for x in re.split("(\w+|[,.])", elem) if x.strip() != ""] for elem in data]
        max_length = max(len(x) for x in data)
        answer = np.zeros(shape=(len(data), max_length, self.dim), dtype=int)
        for i, sent in enumerate(data):
            for j, word in enumerate(sent):
                answer[i, j][self._get_word_indexes(word)] = 1
        return answer


@register("dictionary_vectorizer")
class DictionaryVectorizer(WordIndexVectorizer):
    """
    Transforms words into 0-1 vector of its possible tags, read from a vocabulary file.
    The format of the vocabulary must be word<TAB>tag_1<SPACE>...<SPACE>tag_k

    Args:
        save_path: path to save the vocabulary,
        load_path: path to the vocabulary(-ies),
        min_freq: minimal frequency of tag to memorize this tag,
        unk_token: unknown token to be yielded for unknown words
    """

    def __init__(self, save_path: str, load_path: Union[str, List[str]],
                 min_freq: int = 1, unk_token: str = None, **kwargs) -> None:
        super().__init__(save_path, load_path, **kwargs)
        self.min_freq = min_freq
        self.unk_token = unk_token
        self.load()

    @property
    def dim(self):
        return len(self._t2i)

    def save(self) -> None:
        """Saves the dictionary to self.save_path"""
        with self.save_path.open("w", encoding="utf8") as fout:
            for word, curr_labels in sorted(self.word_tag_mapping.items()):
                curr_labels = [self._i2t[index] for index in curr_labels]
                curr_labels = [x for x in curr_labels if x != self.unk_token]
                fout.write("{}\t{}".format(word, " ".join(curr_labels)))

    def load(self) -> None:
        """Loads the dictionary from self.load_path"""
        if not isinstance(self.load_path, list):
            self.load_path = [self.load_path]
        for i, path in enumerate(self.load_path):
            if isinstance(path, str):
                self.load_path[i] = pathlib.Path(path)
        labels_by_words = defaultdict(set)
        for infile in self.load_path:
            with infile.open("r", encoding="utf8") as fin:
                for line in fin:
                    line = line.strip()
                    if line.count("\t") != 1:
                        continue
                    word, labels = line.split("\t")
                    labels_by_words[word].update(labels.split())
        self._initialize(labels_by_words)

    def _initialize(self, labels_by_words: Dict):
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

    def _get_word_indexes(self, word: AnyStr):
        return self.word_tag_mapping[word]


@register("pymorphy_vectorizer")
class PymorphyVectorizer(WordIndexVectorizer):
    """
    Transforms russian words into 0-1 vector of its possible Universal Dependencies tags.
    Tags are obtained using Pymorphy analyzer (pymorphy2.readthedocs.io)
    and transformed to UD2.0 format using russian-tagsets library (https://github.com/kmike/russian-tagsets).
    All UD2.0 tags that are compatible with produced tags are memorized.
    The list of possible Universal Dependencies tags is read from a file,
    which contains all the labels that occur in UD2.0 SynTagRus dataset.

    Args:
        save_path: path to save the tags list,
        load_path: path to load the list of tags,
        max_pymorphy_variants: maximal number of pymorphy parses to be used. If -1, all parses are used.
    """

    USELESS_KEYS = ["Abbr"]
    VALUE_MAP = {"Ptan": "Plur", "Brev": "Short"}

    def __init__(self, save_path: str, load_path: str, max_pymorphy_variants: int = -1, **kwargs) -> None:
        super().__init__(save_path, load_path, **kwargs)
        self.max_pymorphy_variants = max_pymorphy_variants
        self.load()
        self.memorized_word_indexes = dict()
        self.memorized_tag_indexes = dict()
        self.analyzer = MorphAnalyzer()
        self.converter = converters.converter('opencorpora-int', 'ud20')

    @property
    def dim(self):
        return len(self._t2i)

    def save(self) -> None:
        """Saves the dictionary to self.save_path"""
        with self.save_path.open("w", encoding="utf8") as fout:
            fout.write("\n".join(self._i2t))

    def load(self) -> None:
        """Loads the dictionary from self.load_path"""
        self._i2t = []
        with self.load_path.open("r", encoding="utf8") as fin:
            for line in fin:
                line = line.strip()
                if line == "":
                    continue
                self._i2t.append(line)
        self._t2i = {tag: i for i, tag in enumerate(self._i2t)}
        self._make_tag_trie()

    def _make_tag_trie(self):
        self._nodes = [defaultdict(dict)]
        self._start_nodes_for_pos = dict()
        self._data = [None]
        for tag, code in self._t2i.items():
            pos, tag = make_pos_and_tag(tag, sep=",", return_mode="sorted_items")
            start = self._start_nodes_for_pos.get(pos)
            if start is None:
                start = self._start_nodes_for_pos[pos] = len(self._nodes)
                self._nodes.append(defaultdict(dict))
                self._data.append(None)
            for key, value in tag:
                values_dict = self._nodes[start][key]
                child = values_dict.get(value)
                if child is None:
                    child = values_dict[value] = len(self._nodes)
                    self._nodes.append(defaultdict(dict))
                    self._data.append(None)
                start = child
            self._data[start] = code
        return self

    def find_compatible(self, tag: str) -> List[int]:
        """
        Transforms a Pymorphy tag to a list of indexes of compatible UD tags.

        Args:
            tag: input Pymorphy tag

        Returns:
            indexes of compatible UD tags
        """
        if " " in tag and "_" not in tag:
            pos, tag = tag.split(" ", maxsplit=1)
            tag = sorted([tuple(elem.split("=")) for elem in tag.split("|")])
        else:
            pos, tag = tag.split()[0], []
        if pos not in self._start_nodes_for_pos:
            return []
        tag = [(key, self.VALUE_MAP.get(value, value)) for key, value in tag
               if key not in self.USELESS_KEYS]
        if len(tag) > 0:
            curr_nodes = [(0, self._start_nodes_for_pos[pos])]
            final_nodes = []
        else:
            final_nodes = [self._start_nodes_for_pos[pos]]
            curr_nodes = []
        while len(curr_nodes) > 0:
            i, node_index = curr_nodes.pop()
            # key, value = tag[i]
            node = self._nodes[node_index]
            if len(node) == 0:
                final_nodes.append(node_index)
            for curr_key, curr_values_dict in node.items():
                curr_i, curr_node_index = i, node_index
                while curr_i < len(tag) and tag[curr_i][0] < curr_key:
                    curr_i += 1
                if curr_i == len(tag):
                    final_nodes.extend(curr_values_dict.values())
                    continue
                key, value = tag[curr_i]
                if curr_key < key:
                    for child in curr_values_dict.values():
                        curr_nodes.append((curr_i, child))
                else:
                    child = curr_values_dict.get(value)
                    if child is not None:
                        if curr_i < len(tag) - 1:
                            curr_nodes.append((curr_i + 1, child))
                        else:
                            final_nodes.append(child)
        answer = []
        while len(final_nodes) > 0:
            index = final_nodes.pop()
            if self._data[index] is not None:
                answer.append(self._data[index])
            for elem in self._nodes[index].values():
                final_nodes.extend(elem.values())
        return answer

    def _get_word_indexes(self, word):
        answer = self.memorized_word_indexes.get(word)
        if answer is None:
            parse = self.analyzer.parse(word)
            if self.max_pymorphy_variants > 0:
                parse = parse[:self.max_pymorphy_variants]
            tag_indexes = set()
            for elem in parse:
                tag_indexes.update(set(self._get_tag_indexes(elem.tag)))
            answer = self.memorized_word_indexes[word] = list(tag_indexes)
        return answer

    def _get_tag_indexes(self, pymorphy_tag):
        answer = self.memorized_tag_indexes.get(pymorphy_tag)
        if answer is None:
            tag = self.converter(str(pymorphy_tag))
            answer = self.memorized_tag_indexes[pymorphy_tag] = self.find_compatible(tag)
        return answer
