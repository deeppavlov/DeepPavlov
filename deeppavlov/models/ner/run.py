from slotfill import DstcSlotFillingNetwork
from src.ner_network import NerNetwork
from deeppavlov.core.common.registry import _REGISTRY
from deeppavlov.core.commands.infer import interact_model

CONFIG_PATH = 'config.json'

# Download pretrained model
interact_model(CONFIG_PATH)
