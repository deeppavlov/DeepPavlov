from deeppavlov.training.train import train_agent_models, train_model_from_config
from deeppavlov.infering.infer import interact_agent, interact_model

from hcn.babi_dataset_reader import BabiDatasetReader
from hcn.models.hybrid import HybridCodeNetwork
from hcn.models.bow import BoW_encoder
from hcn.models.embedder import UtteranceEmbed
from hcn.models.et import EntityTracker
from hcn.models.at import ActionTracker
from hcn.models.lstm import LSTM
from commutators.random_commutator import RandomCommutator
from dummy_skill.dummy import DummySkill

###### Train and speak to HCN_go_Dummy agent
AGENT_CONFIG_PATH = '../agent_configs/hcn_go_dummy.json'
train_agent_models(AGENT_CONFIG_PATH)
interact_agent(AGENT_CONFIG_PATH)

##### Train and speak to HCN_go skill separately
MODEL_CONFIG_PATH = 'config.json'
train_model_from_config(MODEL_CONFIG_PATH)
interact_model(MODEL_CONFIG_PATH)
