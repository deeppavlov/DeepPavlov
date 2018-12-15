from collections import defaultdict
from heapdict import heapdict
import itertools
from pathlib import Path
import json
from typing import List, Dict, Union, Optional, Tuple

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.commands.utils import expand_path, parse_config
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.params import from_params
from deeppavlov.core.common.registry import get_model, register
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.models.estimator import Estimator
from deeppavlov.dataset_iterators.morphotagger_iterator import MorphoTaggerDatasetIterator
from deeppavlov.models.morpho_tagger.common_tagger import make_pos_and_tag, make_full_UD_tag


@register('multilingual_tag_normalizer')
class MultilingualTagNormalizer(Estimator):

    def __init__(self, save_path : str, load_path: Union[str, List[str]], max_error=2, **kwargs):
        self.max_error = max_error
        super().__init__(save_path, load_path, **kwargs)

    def save(self):
        info = dict()
        for attr in ["feats", "feats_by_pos", "labels", "max_values",
                     "max_error", "pos", "_counts", "_trie", "_trie_counts"]:
            data = getattr(self, attr)
            if attr == "feats":
                data = {key: dict(value) for key, value in data.items()}
            elif attr == "feats_by_pos":
                data = {key: list(value) for key, value in data.items()}
            elif attr == "labels":
                data = sorted(data)
            elif attr == "_trie":
                data = [dict(elem) for elem in data]
            info[attr] = data
        with open(self.save_path, "w", encoding="utf8") as fout:
            json.dump(info, fout)

    def load(self):
        with open(self.load_path, "r", encoding="utf8") as fin:
            info = json.load(fin)
        for attr, data in info.items():
            if attr == "feats":
                data = defaultdict(lambda: defaultdict(int), data)
            elif attr == "feats_by_pos":
                data = defaultdict(set, data)
            elif attr == "labels":
                data = set(data)
            elif attr == "_trie":
                data = [defaultdict(dict, elem) for elem in data]
            setattr(self, attr, data)

    @property
    def nodes_number(self):
        return len(self._trie)

    def fit(self, labels, indexes):
        labels = list(itertools.chain.from_iterable(
            (elem for elem, index in zip(labels, indexes) if index == 0)))
        labels = [make_pos_and_tag(label, sep=",", return_mode="sorted_dict") for label in labels]
        self.labels = set(labels)
        self.pos = sorted(set(elem[0] for elem in labels))
        self.feats_by_pos = defaultdict(set)
        self.feats = defaultdict(lambda: defaultdict(int))
        for pos, tag in labels:
            for key, value in tag:
                self.feats[key][value] += 1
        self.max_values = {feat: max(values.keys(), key=lambda x: values[x])
                           for feat, values in self.feats.items()}
        self._make_trie(labels)
        return self

    def _get_node(self, source, key, value=None):
        curr = self._trie[source].get(key)
        if curr is None:
            curr = self._add_node(source, key, value)
        else:
            if value is not None:
                curr = curr.get(value)
                if curr is None:
                    curr = self._add_node(source, key, value)
        return curr

    def _add_node(self, source, key, value=None):
        if value is None:
            self._trie[source][key] = self.nodes_number
        else:
            self._trie[source][key][value] = self.nodes_number
        self._trie.append(defaultdict(dict))
        self._counts.append(0)
        return self.nodes_number - 1

    def _make_trie(self, labels):
        self._trie = [defaultdict(dict)]
        self._counts = [0]
        for pos, tag in labels:
            curr = self._get_node(0, pos)
            for key, value in tag:
                curr = self._get_node(curr, key, value)
            self._counts[curr] += 1
        self._make_trie_counts()
        return self

    def _make_trie_counts(self):
        self._trie_counts = self._counts[:]
        order = list(self._trie[0].values())
        while(len(order) > 0):
            curr = order.pop()
            order.extend(child for elem in self._trie[curr].values() for child in elem.values())
        for curr in order[::-1]:
            node = self._trie[curr]
            for key, data in node.items():
                for value, child in data.items():
                    self._trie_counts[curr] = max(self._trie_counts[child], self._counts[curr])
        return self

    def transform(self, tag):
        pos, feats = make_pos_and_tag(tag, sep=",", return_mode="sorted_dict")
        if pos not in self.pos:
            return pos
        answer = []
        for key, value in feats:
            if key not in self.feats:
                continue
            elif value not in self.feats[key]:
                value = self.max_values[key]
            answer.append((key, value))
        if (pos, tuple(answer)) not in self.labels:
            new_answer = self._search_trie(pos, answer)
            if new_answer is not None:
                answer = new_answer
        answer = make_full_UD_tag(pos, answer, mode="sorted_dict")
        return answer

    def __call__(self, data):
        answer = [[self.transform(tag) for tag in elem] for elem in data]
        return answer

    def _search_trie(self, pos, feats):
        curr = self._trie[0][pos]
        key = (curr, 0, tuple())
        value = (0, 0)
        agenda = heapdict({key: value})
        final_answer, min_cost = [], None
        while len(agenda) > 0:
            (curr, index, data), (cost, freq) = agenda.popitem()
            if min_cost is not None and (cost, freq) >= min_cost:
                break
            node = self._trie[curr]
            if index == len(feats):
                if self._counts[curr] > 0 and (min_cost is None or (cost, freq) < min_cost):
                    final_answer, min_cost = data, (cost, freq)
            else:
                feat, value = feats[index]
                feat_data = node.get(feat)
                if feat_data is not None:
                    child = feat_data.get(value)
                    if child is not None:
                        new_data = data + (feats[index],)
                        agenda[(child, index+1, new_data)] = (cost, -self._trie_counts[child])
            if cost < self.max_error:
                if index < len(feats):
                    agenda[(curr, index+1, data)]  = (cost+1, -self._trie_counts[curr])
                for other_feat, feat_data in node.items():
                    if index == len(feats) or other_feat < feat:
                        for value, child in feat_data.items():
                            new_data = data + ((other_feat, value),)
                            agenda[(child, index, new_data)] = (cost + 1, -self._trie_counts[child])
        return final_answer


@register('sent_label_splitter')
class SentLabelSplitter(Component):

    def __init__(self, **kwargs):
        pass

    def __call__(self, X: List[Tuple]) -> Tuple[List[Union[List[str], str]], List[int]]:
        answer = [elem[0] for elem in X], [elem[1] for elem in X]
        return answer