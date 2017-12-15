from deeppavlov.core.commands.infer import interact_model
from deeppavlov.core.commands.utils import set_usr_dir

from deeppavlov.models.spellers.models.error_model import ErrorModel
from deeppavlov.vocabs.wiki_100k_dictionary import Wiki100KDictionary
from deeppavlov.dataset_readers.typos_wikipedia import TyposWikipedia
from deeppavlov.dataset_readers.typos_kartaslov import TyposKartaslov
from deeppavlov.datasets.typos_dataset import TyposDataset

from deeppavlov.models.spellers.train import train

USR_DIR = 'USR_DIR'
paths.USR_PATH = USR_DIR

CONFIG_PATH = 'config_en.json'
usr_dir = set_usr_dir(CONFIG_PATH, USR_DIR)

###### Train all models
train(CONFIG_PATH, usr_dir)

##### Speak to a bot
interact_model(CONFIG_PATH)


