from typing import Union, List

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.chainer import Chainer

from deeppavlov.models.morpho_tagger.common import TagOutputPrettifier,\
    LemmatizedOutputPrettifier, DependencyOutputPrettifier


UD_COLUMN_FEAT_MAPPING = {"id": 0, "word": 1, "lemma": 2, "upos": 3, "feats": 5, "head": 6, "deprel": 7}


@register("joint_tagger_parser")
class JointTaggerParser(Component):
    """
    A class to perform joint morphological and syntactic parsing.
    It is just a wrapper that calls the models for tagging and parsing
    and comprises their results in a single output.

    Args:
        tagger: the morphological tagger model (a :class:`~deeppavlov.core.common.chainer.Chainer` instance)
        parser_path: the syntactic parser model (a :class:`~deeppavlov.core.common.chainer.Chainer` instance)
        output_format: the output format, it may be either `ud` (alias: `conllu`) or `json`.
        to_output_string: whether to convert the output to a list of strings

    Attributes:
        tagger: a morphological tagger model (a :class:`~deeppavlov.core.common.chainer.Chainer` instance)
        parser: a syntactic parser model (a :class:`~deeppavlov.core.common.chainer.Chainer` instance)

    """

    def __init__(self, tagger: Chainer, parser: Chainer,
                 output_format: str = "ud", to_output_string: bool = False,
                 *args, **kwargs):
        if output_format not in ["ud", "conllu", "json", "dict"]:
            UserWarning("JointTaggerParser output_format can be only `ud`, `conllu` or `json`. "\
                        "Unknown format: {}, setting the output_format to `ud`.".format(output_format))
            output_format = "ud"
        self.output_format = output_format
        self.to_output_string = to_output_string
        self.tagger = tagger
        self.parser = parser
        self._check_models()

    def _check_models(self):
        tagger_prettifier = self.tagger[-1]
        if not isinstance(tagger_prettifier, (TagOutputPrettifier, LemmatizedOutputPrettifier)):
            raise ValueError("The tagger should output prettified data: last component of the config "
                             "should be either a TagOutputPrettifier or a LemmatizedOutputPrettifier "
                             "instance.")
        if isinstance(tagger_prettifier, TagOutputPrettifier):
            tagger_prettifier.set_format_mode("ud")
        tagger_prettifier.return_string = False
        parser_prettifier = self.parser[-1]
        if not isinstance(parser_prettifier, DependencyOutputPrettifier):
            raise ValueError("The tagger should output prettified data: last component of the config "
                             "should be either a DependencyOutputPrettifier instance.")
        parser_prettifier.return_string = False

    def __call__(self, data: Union[List[str], List[List[str]]])\
            -> Union[List[List[dict]], List[str], List[List[str]]]:
        r"""Parses a batch of sentences.

        Args:
            data: either a batch of tokenized sentences, or a batch of raw sentences

        Returns:
            `answer`, a batch of parsed sentences. A sentence parse is a list of single word parses.
            Each word parse is either a CoNLL-U-formatted string or a dictionary.
            A sentence parse is returned either as is if ``self.to_output_string`` is ``False``,
            or as a single string, where each word parse begins with a new string.

        .. code-block:: python

            >>> from deeppavlov.core.commands.infer import build_model
            >>> model = build_model("ru_syntagrus_joint_parsing")
            >>> batch = ["Девушка пела в церковном хоре.", "У этой задачи есть сложное решение."]
            >>> print(*model(batch), sep="\\n\\n")
                1	Девушка	девушка	NOUN	_	Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing	2	nsubj	_	_
                2	пела	петь	VERB	_	Aspect=Imp|Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act	0	root	_	_
                3	в	в	ADP	_	_	5	case	_	_
                4	церковном	церковный	ADJ	_	Case=Loc|Degree=Pos|Gender=Masc|Number=Sing	5	amod	_	_
                5	хоре	хор	NOUN	_	Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing	2	obl	_	_
                6	.	.	PUNCT	_	_	2	punct	_	_

                1	У	у	ADP	_	_	3	case	_	_
                2	этой	этот	DET	_	Case=Gen|Gender=Fem|Number=Sing	3	det	_	_
                3	задачи	задача	NOUN	_	Animacy=Inan|Case=Gen|Gender=Fem|Number=Sing	4	obl	_	_
                4	есть	быть	VERB	_	Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act	0	root	_	_
                5	сложное	сложный	ADJ	_	Case=Nom|Degree=Pos|Gender=Neut|Number=Sing	6	amod	_	_
                6	решение	решение	NOUN	_	Animacy=Inan|Case=Nom|Gender=Neut|Number=Sing	4	nsubj	_	_
                7	.	.	PUNCT	_	_	4	punct	_	_

            >>> # Dirty hacks to change model parameters in the code, you should do it in the configuration file.
            >>> model["main"].to_output_string = False
            >>> model["main"].output_format = "json"
            >>> for sent_parse in model(batch):
            >>>     for word_parse in sent_parse:
            >>>         print(word_parse)
            >>>     print("")
                {'id': '1', 'word': 'Девушка', 'lemma': 'девушка', 'upos': 'NOUN', 'feats': 'Animacy=Anim|Case=Nom|Gender=Fem|Number=Sing', 'head': '2', 'deprel': 'nsubj'}
                {'id': '2', 'word': 'пела', 'lemma': 'петь', 'upos': 'VERB', 'feats': 'Aspect=Imp|Gender=Fem|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act', 'head':                   '0', 'deprel': 'root'}
                {'id': '3', 'word': 'в', 'lemma': 'в', 'upos': 'ADP', 'feats': '_', 'head': '5', 'deprel': 'case'}
                {'id': '4', 'word': 'церковном', 'lemma': 'церковный', 'upos': 'ADJ', 'feats': 'Case=Loc|Degree=Pos|Gender=Masc|Number=Sing', 'head': '5', 'deprel': 'amod'}
                {'id': '5', 'word': 'хоре', 'lemma': 'хор', 'upos': 'NOUN', 'feats': 'Animacy=Inan|Case=Loc|Gender=Masc|Number=Sing', 'head': '2', 'deprel': 'obl'}
                {'id': '6', 'word': '.', 'lemma': '.', 'upos': 'PUNCT', 'feats': '_', 'head': '2', 'deprel': 'punct'}

                {'id': '1', 'word': 'У', 'lemma': 'у', 'upos': 'ADP', 'feats': '_', 'head': '3', 'deprel': 'case'}
                {'id': '2', 'word': 'этой', 'lemma': 'этот', 'upos': 'DET', 'feats': 'Case=Gen|Gender=Fem|Number=Sing', 'head': '3', 'deprel': 'det'}
                {'id': '3', 'word': 'задачи', 'lemma': 'задача', 'upos': 'NOUN', 'feats': 'Animacy=Inan|Case=Gen|Gender=Fem|Number=Sing', 'head': '4', 'deprel': 'obl'}
                {'id': '4', 'word': 'есть', 'lemma': 'быть', 'upos': 'VERB', 'feats': 'Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act', 'head': '0',                'deprel': 'root'}
                {'id': '5', 'word': 'сложное', 'lemma': 'сложный', 'upos': 'ADJ', 'feats': 'Case=Nom|Degree=Pos|Gender=Neut|Number=Sing', 'head': '6', 'deprel': 'amod'}
                {'id': '6', 'word': 'решение', 'lemma': 'решение', 'upos': 'NOUN', 'feats': 'Animacy=Inan|Case=Nom|Gender=Neut|Number=Sing', 'head': '4', 'deprel': 'nsubj'}
                {'id': '7', 'word': '.', 'lemma': '.', 'upos': 'PUNCT', 'feats': '_', 'head': '4', 'deprel': 'punct'}

        """
        tagger_output = self.tagger(data)
        parser_output = self.parser(data)
        answer = []
        for i, (tagger_sent, parser_sent) in enumerate(zip(tagger_output, parser_output)):
            curr_sent_answer = []
            for j, curr_word_tagger_output in enumerate(tagger_sent):
                curr_word_tagger_output = curr_word_tagger_output.split("\t")
                curr_word_parser_output = parser_sent[j].split("\t")
                curr_word_answer = curr_word_tagger_output[:]
                # setting parser output
                curr_word_answer[6:8] = curr_word_parser_output[6:8]
                if self.output_format in ["json", "dict"]:
                    curr_word_answer = {key: curr_word_answer[index]
                                        for key, index in UD_COLUMN_FEAT_MAPPING.items()}
                    if self.to_output_string:
                        curr_word_answer = str(curr_word_answer)
                elif self.to_output_string:
                    curr_word_answer = "\t".join(curr_word_answer)
                curr_sent_answer.append(curr_word_answer)
            if self.to_output_string:
                curr_sent_answer = "\n".join(str(x) for x in curr_sent_answer)
            answer.append(curr_sent_answer)
        return answer


