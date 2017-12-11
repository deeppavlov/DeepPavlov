from deeppavlov.core.commands.infer import interact_agent, interact_model
from deeppavlov.core.commands.train import train_model_from_config, train_agent_models

from deeppavlov.models.hcn.babi_dataset_reader import BabiDatasetReader
from deeppavlov.models.hcn.models.hybrid import HybridCodeNetwork
from deeppavlov.models.hcn.models.bow import BoW_encoder
from deeppavlov.models.hcn.models.embedder import UtteranceEmbed
from deeppavlov.models.hcn.models.et import EntityTracker
from deeppavlov.models.hcn.models.at import ActionTracker
from deeppavlov.models.hcn.models.lstm import LSTM
from deeppavlov.models.commutators.random_commutator import RandomCommutator
from deeppavlov.models.dummy_skill.dummy import DummySkill

###### Train and speak to HCN_go_Dummy agent
AGENT_CONFIG_PATH = '../agent_configs/hcn_go_dummy.json'
train_agent_models(AGENT_CONFIG_PATH)
interact_agent(AGENT_CONFIG_PATH)

##### Train and speak to HCN_go skill separately
MODEL_CONFIG_PATH = 'config.json'
train_model_from_config(MODEL_CONFIG_PATH)
interact_model(MODEL_CONFIG_PATH)



