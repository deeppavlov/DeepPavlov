from deeppavlov.core.commands.infer import interact_agent
from deeppavlov.core.commands.train import train_agent_models
from deeppavlov.core.commands.utils import set_usr_dir

###### Train and speak to HCN_go_Dummy agent
AGENT_CONFIG_PATH = 'hcn_go_dummy.json'
usr_dir = set_usr_dir(AGENT_CONFIG_PATH)
# train_agent_models(AGENT_CONFIG_PATH)
interact_agent(AGENT_CONFIG_PATH)