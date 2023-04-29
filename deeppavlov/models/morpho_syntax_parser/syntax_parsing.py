# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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

from typing import List, Optional, Tuple, Union

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component


def make_pos_and_tag(tag: str, sep: str = ",",
                     return_mode: Optional[str] = None) -> Tuple[str, Union[str, list, dict, tuple]]:
    """
    Args:
        tag: the part-of-speech tag
        sep: the separator between part-of-speech tag and grammatical features
        return_mode: the type of return value, can be None, list, dict or sorted_items
    Returns:
        the part-of-speech label and grammatical features in required format
    """
    if tag.endswith(" _"):
        tag = tag[:-2]
    if sep in tag:
        pos, tag = tag.split(sep, maxsplit=1)
    else:
        pos, tag = tag, ("_" if return_mode is None else "")
    if return_mode in ["dict", "list", "sorted_items"]:
        tag = tag.split("|") if tag != "" else []
        if return_mode in ["dict", "sorted_items"]:
            tag = dict(tuple(elem.split("=")) for elem in tag)
            if return_mode == "sorted_items":
                tag = tuple(sorted(tag.items()))
    return pos, tag


class OutputPrettifier(Component):
    """Base class for formatting the output of dependency parser and morphotagger"""

    def __init__(self, return_string: bool = True, begin: str = "", end: str = "\n", sep: str = "\n",
                 **kwargs) -> None:
        self.return_string = return_string
        self.begin = begin
        self.end = end
        self.sep = sep

    def prettify(self, tokens: List[str], heads: List[int], deps: List[str]) -> Union[List[str], str]:
        raise NotImplementedError

    def __call__(self, X: List[List[str]], Y: List[List[int]], Z: List[List[str]]) -> List[Union[List[str], str]]:
        """Calls the :meth:`~prettify` function for each input sentence.
        Args:
            X: a list of input sentences
            Y: a list of lists of head positions for sentence words
            Z: a list of lists of dependency labels for sentence words
        Returns:
            a list of prettified UD outputs
        """
        return [self.prettify(x, y, z) for x, y, z in zip(X, Y, Z)]


@register('dependency_output_prettifier')
class DependencyOutputPrettifier(OutputPrettifier):
    """Class which prettifies dependency parser output
    to 10-column (Universal Dependencies) format.
    Args:
        begin: a string to append in the beginning
        end: a string to append in the end
        sep: separator between word analyses
    """

    def __init__(self, return_string: bool = True, begin: str = "", end: str = "\n", sep: str = "\n",
                 **kwargs) -> None:
        super().__init__(return_string, begin, end, sep, **kwargs)
        self.format_string = "{}\t{}\t_\t_\t_\t_\t{}\t{}\t_\t_"

    def prettify(self, tokens: List[str], heads: List[int], deps: List[str]) -> Union[List[str], str]:
        """Prettifies output of dependency parser.
        Args:
            tokens: tokenized source sentence
            heads: list of head positions, the output of the parser
            deps: list of head positions, the output of the parser
        Returns:
            the prettified output of the parser
        """
        answer = []
        for i, (word, head, dep) in enumerate(zip(tokens, heads, deps)):
            answer.append(self.format_string.format(i + 1, word, head, dep))
        if self.return_string:
            answer = self.begin + self.sep.join(answer) + self.end
        return answer


@register('lemmatized_output_prettifier')
class LemmatizedOutputPrettifier(OutputPrettifier):
    """Class which prettifies morphological tagger output to 4-column
    or 10-column (Universal Dependencies) format.
    Args:
        format_mode: output format,
            in `basic` mode output data contains 4 columns (id, word, pos, features),
            in `conllu` or `ud` mode it contains 10 columns:
            id, word, lemma, pos, xpos, feats, head, deprel, deps, misc
            (see http://universaldependencies.org/format.html for details)
            Only id, word, lemma, tag and pos columns are predicted in current version,
            other columns are filled by `_` value.
        begin: a string to append in the beginning
        end: a string to append in the end
        sep: separator between word analyses
    """

    def __init__(self, return_string: bool = True, begin: str = "", end: str = "\n", sep: str = "\n",
                 **kwargs) -> None:
        super().__init__(return_string, begin, end, sep, **kwargs)
        self.format_string = "{}\t{}\t{}\t{}\t_\t{}\t_\t_\t_\t_"

    def prettify(self, tokens: List[str], tags: List[str], lemmas: List[str]) -> Union[List[str], str]:
        """Prettifies output of morphological tagger.
        Args:
            tokens: tokenized source sentence
            tags: list of tags, the output of a tagger
            lemmas: list of lemmas, the output of a lemmatizer
        Returns:
            the prettified output of the tagger.
        Examples:
            >>> sent = "John really likes pizza .".split()
            >>> tags = ["PROPN,Number=Sing", "ADV",
            >>>         "VERB,Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin",
            >>>         "NOUN,Number=Sing", "PUNCT"]
            >>> lemmas = "John really like pizza .".split()
            >>> prettifier = LemmatizedOutputPrettifier()
            >>> self.prettify(sent, tags, lemmas)
                1	John	John	PROPN	_	Number=Sing	_	_	_	_
                2	really	really	ADV	_	_	_	_	_	_
                3	likes	like	VERB	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	_	_	_	_
                4	pizza	pizza	NOUN	_	Number=Sing	_	_	_	_
                5	.	.	PUNCT	_	_	_	_	_	_
        """
        answer = []
        for i, (word, tag, lemma) in enumerate(zip(tokens, tags, lemmas)):
            pos, tag = make_pos_and_tag(tag, sep=",")
            answer.append(self.format_string.format(i + 1, word, lemma, pos, tag))
        if self.return_string:
            answer = self.begin + self.sep.join(answer) + self.end
        return answer
