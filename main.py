from deeppavlov.core.commands.infer import build_model_from_config
from deeppavlov.core.commands.train import train_model_from_config
from deeppavlov.dataset_readers.squad_dataset_reader import SquadDatasetReader
from deeppavlov.datasets.squad_dataset import SquadDataset

import json

d = SquadDatasetReader()
data = d.read('download/squad')
print(data.keys())

chainer = train_model_from_config('deeppavlov/configs/squad/squad.json')

dataset = SquadDataset(data, seed=None, shuffle=True)
bg = dataset.batch_generator(5, data_type='train')

els = [el for el in bg]

for x, y in els[:3]:
    print(x)
    out = chainer(x, y)
    #print(out.shape)
    for el in out:
        print(el)

x, y = dataset.iter_all()
