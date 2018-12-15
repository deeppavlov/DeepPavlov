from deeppavlov import build_model, configs
from deeppavlov.dataset_readers.morphotagging_dataset_reader import read_infile
from deeppavlov.models.morpho_tagger.common import call_model

data = read_infile("/home/alexeysorokin/data/Data/UD2.0_test/ru_syntagrus.conllu")[:10]
data, tags = [elem[0] for elem in data], [elem[1] for elem in data]
model = build_model(configs.morpho_tagger.UD2_0.morpho_ru_syntagrus_pymorphy_lemmatize, download=True)
print(call_model(model, data)[0])