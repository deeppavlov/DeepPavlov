from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import model as get_model
from deeppavlov.core.common.params import from_params
from deeppavlov.core.data.dataset import Dataset
from deeppavlov.core.commands.train import fit_chainer, _train_batches, train_model_from_config
from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.common.metrics_registry import get_metrics_by_names

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODE = "train"

def test_reader(config):
    reader_config = config['dataset_reader']
    reader = get_model(reader_config['name'])()
    data_path = reader_config.get('data_path', '')
    read_params = reader_config.get("read_params", dict())
    return reader.read(data_path, **read_params)


if __name__ == "__main__":
    if MODE == "dataset_reader":
        filenames = ["test_reader", "test_single_file_reader"]
        for filename in filenames:
            config_path = "configs/morpho_tagger/{}.json".format(filename)
            config = read_json(config_path)
            print(config_path)
            data = test_reader(config)
            print("train dataset length: {}".format(len(data['train'])))
            print("dev dataset length: {}".format(len(data['valid'])))
    elif MODE == "dataset":
        config = read_json("configs/morpho_tagger/test_dataset.json")
        data = test_reader(config)
        dataset: Dataset = from_params(config['dataset'], data=data)
        print("Train batch generation:")
        for i, batch in enumerate(dataset.batch_generator(
                batch_size=config["train_batch_size"], data_type="train"), 1):
            print("Train batch {}, {} items, lengths from {} to {}".format(
                i, len(batch[0]), len(batch[0][0]), len(batch[0][-1])))
        print("Dev batch generation:")
        for i, batch in enumerate(dataset.batch_generator(
                batch_size=config["dev_batch_size"], data_type="valid"), 1):
            print("Dev batch {}, {} items, lengths from {} to {}".format(
                i, len(batch[0]), len(batch[0][0]), len(batch[0][-1])))
    elif MODE == "train":
        # config = read_json("configs/morpho_tagger/train_config.json")
        # data = test_reader(config)
        # dataset = from_params(config['dataset'], data=data)
        # model = fit_chainer(config, dataset)
        # train_config = { 'metrics': ['accuracy'], 'validate_best': True, 'test_best': True}
        # train_config.update(config["train"])
        # metrics_functions = list(zip(train_config['metrics'],
        #                              get_metrics_by_names(train_config['metrics'])))
        # _train_batches(model, dataset, train_config, metrics_functions)
        train_model_from_config("configs/morpho_tagger/test_config.json", is_trained=True)
