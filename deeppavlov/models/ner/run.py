from deeppavlov.core.commands.infer import interact_model
from .slotfill import DstcSlotFillingNetwork
from .ner_network import NerNetwork
from .model_loader import load_ner_dstc_model

CONFIG_PATH = 'config.json'

# Download pretrained model
load_ner_dstc_model('model/')
interact_model(CONFIG_PATH)
