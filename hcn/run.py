from deeppavlov.training.train import train_model_from_config

from hcn.babi_dataset_reader import BabiDatasetReader
from hcn.models.hybrid import HybridCodeNetwork
from hcn.models.bow import BoW_encoder
from hcn.models.embedder import UtteranceEmbed
from hcn.models.et import EntityTracker
from hcn.models.at import ActionTracker
from hcn.models.lstm import LSTM

CONFIG_PATH = 'config.json'
USR_DIR = 'USR_DIR'
train_model_from_config(CONFIG_PATH, USR_DIR)
