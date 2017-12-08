from ner.slotfill import DstcSlotFillingNetwork
from deeppavlov.common.registry import _REGISTRY
from deeppavlov.infering.infer import interact
from deeppavlov.data.utils import download_untar


CONFIG_PATH = 'config.json'

# Download pretrained model
download_untar('http://lnsigo.mipt.ru/export/ner_dstc_model.tar.gz', 'model/')
interact(CONFIG_PATH)

