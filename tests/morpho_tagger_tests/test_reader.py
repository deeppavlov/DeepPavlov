from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import model as get_model
from deeppavlov.core.common.params import from_params
from deeppavlov.core.data.dataset import Dataset

MODE = "dataset"

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
        dataset : Dataset = from_params(config['dataset'], data=data)
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
