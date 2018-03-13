import sys
import getopt

from pathlib import Path


from deeppavlov.core.commands.utils import expand_path, set_deeppavlov_root
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.params import from_params
from deeppavlov.core.common.registry import model as get_model
from deeppavlov.core.data.dataset import Dataset

from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.models.morpho_tagger.common import make_pos_and_tag

SHORT_OPTS = "Tp"


def predict_with_model(config_path):
    config = read_json(config_path)
    set_deeppavlov_root(config)

    reader_config = config['dataset_reader']
    reader = get_model(reader_config['name'])()
    data_path = expand_path(reader_config.get('data_path', ''))
    read_params = reader_config.get("read_params", dict())
    data = reader.read(data_path, **read_params)

    dataset_config = config['dataset']
    dataset: Dataset = from_params(dataset_config, data=data)

    model = build_model_from_config(config, load_trained=True)
    answers = [None] * len(dataset.test)
    batch_size = config['predict'].get("batch_size", -1)
    for indexes, (x, _) in dataset.batch_generator(
            batch_size=batch_size, data_type="test", shuffle=False, return_indexes=True):
        y = model(x)
        for i, elem in zip(indexes, y):
            answers[i] = elem
    outfile = config['predict'].get("outfile")
    # outfile_path = Path(outfile).parent
    # outfile_path.mkdir(exist_ok=True)
    if outfile is not None:
        with open(outfile, "w", encoding="utf8") as fout:
            for (sent, _), tags in zip(data['test'], answers):
                for i, (word, tag) in enumerate(zip(sent, tags)):
                    fout.write("{}\t{}\t{}\n".format(i+1, word, *make_pos_and_tag(tag)))
                fout.write("\n")

    return answers


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    mode = "train"
    for opt, val in opts:
        if opt in ["-T"]:
            mode = "test"
        elif opt in ["-p"]:
            mode = "predict"
        else:
            raise ValueError("Unknown option {}".format(opt))
    config_path = args[0]
    if mode == "predict":
        predict_with_model(config_path)
    elif mode == "test":
        train_model_from_config(config_path, is_trained=True)
    elif mode == "train":
        train_model_from_config(config_path, is_trained=False)