from pathlib import Path
from typing import List, Dict, Union, Optional

from deeppavlov.core.commands.infer import build_model
from deeppavlov.core.commands.utils import expand_path, parse_config
from deeppavlov.core.common.params import from_params
from deeppavlov.core.common.registry import get_model
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.dataset_iterators.morphotagger_iterator import MorphoTaggerDatasetIterator
from deeppavlov.models.morpho_tagger.common_tagger import make_pos_and_tag


def predict_with_model(config_path: [Path, str]) -> List[Optional[List[str]]]:
    """Returns predictions of morphotagging model given in config :config_path:.

    Args:
        config_path: a path to config

    Returns:
        a list of morphological analyses for each sentence. Each analysis is either a list of tags
        or a list of full CONLL-U descriptions.

    """
    config = parse_config(config_path)

    reader_config = config['dataset_reader']
    reader = get_model(reader_config['class_name'])()
    data_path = expand_path(reader_config.get('data_path', ''))
    read_params = {k: v for k, v in reader_config.items() if k not in ['class_name', 'data_path']}
    data: Dict = reader.read(data_path, **read_params)

    iterator_config = config['dataset_iterator']
    iterator: MorphoTaggerDatasetIterator = from_params(iterator_config, data=data)

    model = build_model(config, load_trained=True)
    answers = [None] * len(iterator.test)
    batch_size = config['predict'].get("batch_size", -1)
    for indexes, (x, _) in iterator.gen_batches(
            batch_size=batch_size, data_type="test", shuffle=False, return_indexes=True):
        y = model(x)
        for i, elem in zip(indexes, y):
            answers[i] = elem
    outfile = config['predict'].get("outfile")
    if outfile is not None:
        outfile = Path(outfile)
        if not outfile.exists():
            outfile.parent.mkdir(parents=True, exist_ok=True)
        with open(outfile, "w", encoding="utf8") as fout:
            for elem in answers:
                fout.write(elem + "\n")
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
            self.format_string =  "{}\t{}\t{}\t{}"
        elif self.format_mode in ["conllu", "ud"]:
            self.format_string = "{}\t{}\t_\t{}\t_\t{}\t_\t_\t_\t_"
        else:
            raise ValueError("Wrong mode for TagOutputPrettifier: {}, "
                             "it must be 'basic', 'conllu' or 'ud'.".format(self.mode))

    def __call__(self, X: List[List[str]], Y: List[List[str]]) -> List[Union[List[str], str]]:
        """Calls the ``prettify`` function for each input sentence.

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
