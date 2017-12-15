from slotfill import DstcSlotFillingNetwork
from src.ner_network import NerNetwork
from deeppavlov.core.common.registry import _REGISTRY
from deeppavlov.core.commands.infer import interact_model
from model_loader import load_ner_dstc_model

CONFIG_PATH = 'config.json'

# Download pretrained model
load_ner_dstc_model('model/')
interact_model(CONFIG_PATH)
