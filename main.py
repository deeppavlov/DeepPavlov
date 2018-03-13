from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.dataset_readers.squad_dataset_reader import SquadDatasetReader
from deeppavlov.datasets.squad_dataset import SquadDataset

import json

d = SquadDatasetReader()
data = d.read('download/squad')
print(data.keys())

chainer = build_model_from_config(json.load(open('deeppavlov/configs/squad/squad.json')))

dataset = SquadDataset(data, seed=None, shuffle=True)
bg = dataset.batch_generator(3, data_type='train')

els = [el for el in bg]

for x, y in els[:3]:
    print(x)
    y_predicted = chainer(x)
    print(y_predicted)
