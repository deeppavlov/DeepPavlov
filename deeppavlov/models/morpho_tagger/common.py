import re
from pathlib import Path

from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.commands.utils import set_deeppavlov_root, expand_path
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.params import from_params
from deeppavlov.core.common.registry import model as get_model
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component

from deeppavlov.dataset_iterators.morphotagger_iterator import MorphoTaggerDatasetIterator
from deeppavlov.models.morpho_tagger.common_tagger import make_pos_and_tag



def predict_with_model(config_path):
    config = read_json(config_path)
    set_deeppavlov_root(config)

    reader_config = config['dataset_reader']
    reader = get_model(reader_config['name'])()
    data_path = expand_path(reader_config.get('data_path', ''))
    read_params = {k: v for k, v in reader_config.items() if k not in ['name', 'data_path']}
    data = reader.read(data_path, **read_params)

    iterator_config = config['dataset_iterator']
    iterator: MorphoTaggerDatasetIterator =\
        from_params(iterator_config, data=data)

    model = build_model_from_config(config, load_trained=True)
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


def prettify(sent, tags, return_string=True, begin="",  end="", sep="\n"):
    """

    x: str, sentence
    y: list, a sequence of tags

    x = "John likes, really likes pizza"
    y = ["NNP", "VBZ", "PUNCT", "RB", "VBZ", "NN"]

    answer:
        1  John    NNP
        2  likes   VBZ
        3  ,   PUNCT
        4  really  RB
        5  likes   VBZ
        6  pizza   NN
        7  .    SENT
    """
    if isinstance(sent, str):
        words = [x for x in re.split("(\w+|[,.])", sent) if x.strip() != ""]
    else:
        words = sent
    answer = []
    for i, (word, tag) in enumerate(zip(words, tags)):
        answer.append("{}\t{}\t{}\t{}".format(i+1, word, *make_pos_and_tag(tag)))
    if return_string:
        answer = begin + sep.join(answer) + end
    return answer


@register('tag_output_prettifier')
class TagOutputPrettifier(Component):

    def __init__(self, return_string=True, begin="", end="", sep="\n", *args, **kwargs):
        self.return_string = return_string
        self.begin = begin
        self.end = end
        self.sep = sep

    def __call__(self, X, Y):
        return [prettify(x, y, return_string=self.return_string,
                         begin=self.begin, end=self.end, sep=self.sep)
                for x, y in zip(X, Y)]
