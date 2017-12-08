from deeppavlov.training.train import train_model_from_config
from deeppavlov.infering.infer import interact

from hcn.babi_dataset_reader import BabiDatasetReader
from hcn.models.hybrid import HybridCodeNetwork
from hcn.models.bow import BoW_encoder
from hcn.models.embedder import UtteranceEmbed
from hcn.models.et import EntityTracker
from hcn.models.at import ActionTracker
from hcn.models.lstm import LSTM

CONFIG_PATH = 'config.json'
###### Train all models
train_model_from_config(CONFIG_PATH)

##### Speak to a bot
interact(CONFIG_PATH)


