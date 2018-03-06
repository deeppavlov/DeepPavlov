from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.registry import model as get_model


def test_reader(config_path: str):
    config = read_json(config_path)
    reader_config = config['dataset_reader']
    reader = get_model(reader_config['name'])()
    data_path = reader_config.get('data_path', '')
    read_params = reader_config.get("read_params", dict())
    data = reader.read(data_path, **read_params)
    print("train dataset length: {}".format(len(data['train'])))
    print("dev dataset length: {}".format(len(data['valid'])))
    return


if __name__ == "__main__":
    filenames = ["test_reader", "test_single_file_reader"]
    for filename in filenames:
        filename = "configs/morpho_tagger/{}.json".format(filename)
        print(filename)
        test_reader(filename)