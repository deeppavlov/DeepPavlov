from deeppavlov.models.ner.slotfill import DstcSlotFillingNetwork
from deeppavlov.models.ner.ner_network import NerNetwork
from deeppavlov.datasets.dstc2_datasets import DstcNerDataset
from deeppavlov.dataset_readers.dstc2_dataset_reader import DSTC2DatasetReader

from deeppavlov.core.commands.infer import interact_model
from deeppavlov.core.commands.train import train_model_from_config

CONFIG_PATH = 'config.json'
train_model_from_config(CONFIG_PATH)
interact_model(CONFIG_PATH)
