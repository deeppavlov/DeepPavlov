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

import sys
from pathlib import Path
from typing import List, Union, Optional

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.commands.utils import expand_path, parse_config
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.dataset_readers.morphotagging_dataset_reader import read_infile
from deeppavlov.models.morpho_tagger.common_tagger import make_pos_and_tag


def predict_with_model(config_path: [Path, str], infile: Optional[Union[Path, str]] = None,
                       input_format: str = "ud", batch_size: [int] = 16,
                       output_format: str = "basic") -> List[Optional[List[str]]]:
    """Returns predictions of morphotagging model given in config :config_path:.

    Args:
        config_path: a path to config

    Returns:
        a list of morphological analyses for each sentence. Each analysis is either a list of tags
        or a list of full CONLL-U descriptions.

    """
    config = parse_config(config_path)
    if infile is None:
        if sys.stdin.isatty():
            raise RuntimeError('To process data from terminal please use interact mode')
        infile = sys.stdin
    else:
        infile = expand_path(infile)
    if input_format in ["ud", "conllu", "vertical"]:
        from_words = (input_format == "vertical")
        data: List[tuple] = read_infile(infile, from_words=from_words)
        # keeping only sentences
        data = [elem[0] for elem in data]
    else:
        if infile is not sys.stdin:
            with open(infile, "r", encoding="utf8") as fin:
                data = fin.readlines()
        else:
            data = sys.stdin.readlines()
    model = build_model(config, load_trained=True)
    for elem in model.pipe:
        if isinstance(elem[-1], TagOutputPrettifier):
            elem[-1].set_format_mode(output_format)
    answers = model.batched_call(data, batch_size=batch_size)
    return answers


@register('tag_output_prettifier')
class TagOutputPrettifier(Component):
    """Class which prettifies morphological tagger output to 4-column
    or 10-column (Universal Dependencies) format.

    Args:
        format_mode: output format,
            in `basic` mode output data contains 4 columns (id, word, pos, features),
            in `conllu` or `ud` mode it contains 10 columns:
            id, word, lemma, pos, xpos, feats, head, deprel, deps, misc
            (see http://universaldependencies.org/format.html for details)
            Only id, word, tag and pos values are present in current version,
            other columns are filled by `_` value.
        return_string: whether to return a list of strings or a single string
        begin: a string to append in the beginning
        end: a string to append in the end
        sep: separator between word analyses
    """

    def __init__(self, format_mode: str = "basic", return_string: bool = True,
                 begin: str = "", end: str = "", sep: str = "\n", **kwargs) -> None:
        self.set_format_mode(format_mode)
        self.return_string = return_string
        self.begin = begin
        self.end = end
        self.sep = sep

    def set_format_mode(self, format_mode: str = "basic") -> None:
        """A function that sets format for output and recalculates `self.format_string`.

        Args:
            format_mode: output format,
                in `basic` mode output data contains 4 columns (id, word, pos, features),
                in `conllu` or `ud` mode it contains 10 columns:
                id, word, lemma, pos, xpos, feats, head, deprel, deps, misc
                (see http://universaldependencies.org/format.html for details)
                Only id, word, tag and pos values are present in current version,
                other columns are filled by `_` value.

        Returns:
        """
        self.format_mode = format_mode
        self._make_format_string()

    def _make_format_string(self) -> None:
        if self.format_mode == "basic":
            self.format_string = "{}\t{}\t{}\t{}"
        elif self.format_mode.lower() in ["conllu", "ud"]:
            self.format_string = "{}\t{}\t_\t{}\t_\t{}\t_\t_\t_\t_"
        else:
            raise ValueError("Wrong mode for TagOutputPrettifier: {}, "
                             "it must be 'basic', 'conllu' or 'ud'.".format(self.mode))

    def __call__(self, X: List[List[str]], Y: List[List[str]]) -> List[Union[List[str], str]]:
        """Calls the :meth:`~prettify` function for each input sentence.

        Args:
            X: a list of input sentences
            Y: a list of list of tags for sentence words

        Returns:
            a list of prettified morphological analyses
        """
        return [self.prettify(x, y) for x, y in zip(X, Y)]

    def prettify(self, tokens: List[str], tags: List[str]) -> Union[List[str], str]:
        """Prettifies output of morphological tagger.

        Args:
            tokens: tokenized source sentence
            tags: list of tags, the output of a tagger

        Returns:
            the prettified output of the tagger.

        Examples:
            >>> sent = "John really likes pizza .".split()
            >>> tags = ["PROPN,Number=Sing", "ADV",
            >>>         "VERB,Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin",
            >>>         "NOUN,Number=Sing", "PUNCT"]
            >>> prettifier = TagOutputPrettifier(mode='basic')
            >>> self.prettify(sent, tags)
                1	John	PROPN	Number=Sing
                2	really	ADV	_
                3	likes	VERB	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin
                4	pizza	NOUN	Number=Sing
                5	.	PUNCT	_
            >>> prettifier = TagOutputPrettifier(mode='ud')
            >>> self.prettify(sent, tags)
                1	John	_	PROPN	_	Number=Sing	_	_	_	_
                2	really	_	ADV	_	_	_	_	_	_
                3	likes	_	VERB	_	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	_	_	_	_
                4	pizza	_	NOUN	_	Number=Sing	_	_	_	_
                5	.	_	PUNCT	_	_	_	_	_	_
        """
        answer = []
        for i, (word, tag) in enumerate(zip(tokens, tags)):
            answer.append(self.format_string.format(i + 1, word, *make_pos_and_tag(tag)))
        if self.return_string:
            answer = self.begin + self.sep.join(answer) + self.end
        return answer


@register('lemmatized_output_prettifier')
class LemmatizedOutputPrettifier(Component):
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
        return_string: whether to return a list of strings or a single string
        begin: a string to append in the beginning
        end: a string to append in the end
        sep: separator between word analyses
    """

    def __init__(self, return_string: bool = True,
                 begin: str = "", end: str = "", sep: str = "\n", **kwargs) -> None:
        self.return_string = return_string
        self.begin = begin
        self.end = end
        self.sep = sep
        self.format_string = "{0}\t{1}\t{4}\t{2}\t_\t{3}\t_\t_\t_\t_"

    def __call__(self, X: List[List[str]], Y: List[List[str]], Z: List[List[str]]) -> List[Union[List[str], str]]:
        """Calls the :meth:`~prettify` function for each input sentence.

        Args:
            X: a list of input sentences
            Y: a list of list of tags for sentence words
            Z: a list of lemmatized sentences

        Returns:
            a list of prettified morphological analyses
        """
        return [self.prettify(*elem) for elem in zip(X, Y, Z)]

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
            answer.append(self.format_string.format(i + 1, word, pos, tag, lemma))
        if self.return_string:
            answer = self.begin + self.sep.join(answer) + self.end
        return answer


@register('dependency_output_prettifier')
class DependencyOutputPrettifier(Component):
    """Class which prettifies dependency parser output
    to 10-column (Universal Dependencies) format.

    Args:
        return_string: whether to return a list of strings or a single string
        begin: a string to append in the beginning
        end: a string to append in the end
        sep: separator between word analyses
    """

    def __init__(self, return_string: bool = True, begin: str = "",
                 end: str = "", sep: str = "\n", **kwargs) -> None:
        self.return_string = return_string
        self.begin = begin
        self.end = end
        self.sep = sep
        self.format_string = "{}\t{}\t_\t_\t_\t_\t{}\t{}\t_\t_"

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
