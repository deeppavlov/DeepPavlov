from ner.slotfill import DstcSlotFillingNetwork
from ner.src.ner_network import NerNetwork
from deeppavlov.common.registry import _REGISTRY
from deeppavlov.infering.infer import interact
from ner.model_loader import load_ner_dstc_model

CONFIG_PATH = 'config.json'

# Download pretrained model
load_ner_dstc_model('model/')
interact(CONFIG_PATH)
