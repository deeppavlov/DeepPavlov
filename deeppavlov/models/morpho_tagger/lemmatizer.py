import pathlib
from collections import defaultdict
import re
from typing import List, Dict, Generator, Tuple, Any, AnyStr, Union, Optional
from abc import abstractmethod
import numpy as np

from pymorphy2 import MorphAnalyzer
from russian_tagsets import converters

from deeppavlov.core.models.serializable import Serializable
from deeppavlov.core.common.registry import register
from deeppavlov.models.morpho_tagger.common_tagger import make_pos_and_tag, get_tag_distance


class BasicLemmatizer(Serializable):
    """
    A basic class for lemmatizers.
    """

    def __init__(self, save_path: Optional[str] = None,
                 load_path: Optional[str] = None, **kwargs) -> None:
        super().__init__(save_path, load_path, **kwargs)

    @abstractmethod
    def _lemmatize(self, word: str, tag: Optional[str] = None) -> str:
        """
        Lemmatizes a separate word given its tag.

        Args:
            word: the inpurt word.
            tag: optional morphological tag.

        Returns:
            a lemmatized word
        """
        raise NotImplementedError("Your lemmatizer must implement the abstract method _lemmatize.")

    def __call__(self, data: List[List[str]], tags: Optional[List[List[str]]] = None) -> List[List[str]]:
        """
        Lemmatizes each word in a batch of sentences.

        Args:
            data: the batch of sentences (lists of words).
            tags: the batch of morphological tags (if available).

        Returns:
            a batch of lemmatized sentences.
        """
        if tags is None:
            tags = [[None for _ in sent] for sent in data]
        if len(tags) != len(data):
            raise ValueError("There must be the same number of tag sentences as the number of word sentences.")
        if any((len(elem[0]) != len(elem[1])) for elem in zip(data, tags)):
            raise ValueError("Tag sentence must be of the same length as the word sentence.")
        answer = [[self._lemmatize(word, tag) for word, tag in zip(*elem)] for elem in zip(data, tags)]
        return answer

@register("UD_pymorphy_lemmatizer")
class UDPymorphyLemmatizer(BasicLemmatizer):
    """
    A class that returns a normal form of a Russian word given its morphological tag in UD format.
    Lemma is selected from one of PyMorphy parses,
    the parse whose tag resembles the most a known UD tag is chosen.
    """
    def __init__(self, save_path: Optional[str] = None, load_path: Optional[str] = None,
                 transform_lemmas=False, **kwargs) -> None:
        self.transform_lemmas = transform_lemmas
        self._reset()
        self.analyzer = MorphAnalyzer()
        self.converter = converters.converter("opencorpora-int", "ud20")
        super().__init__(save_path, load_path, **kwargs)

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

    def _reset(self):
        self.memo = dict()

    def _lemmatize(self, word: str, tag: Optional[str] = None):
        lemma = self.memo.get((word, tag))
        if lemma is not None:
            return lemma
        parses = self.analyzer.parse(word)
        best_lemma, best_distance = word, np.inf
        for i, parse in enumerate(parses):
            curr_tag, curr_lemma = self.converter(str(parse.tag)), parse.normal_form
            distance = get_tag_distance(tag, curr_tag)
            if distance < best_distance:
                best_lemma, best_distance = curr_lemma, distance
                if distance == 0:
                    break
        self.memo[(word, tag)] = best_lemma
        return best_lemma



