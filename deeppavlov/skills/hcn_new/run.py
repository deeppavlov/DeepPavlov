#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deeppavlov.core.commands.train import train_agent_models, train_model_from_config
from deeppavlov.core.commands.infer import interact_agent, interact_model

from deeppavlov.skills.ner.slotfill import DstcSlotFillingNetwork
from deeppavlov.core.data.dataset_readers.dstc2_dataset_reader import DSTC2DatasetReader

from dstc2_dataset import DSTC2Dataset
from models.hcn import HybridCodeNetworkBot

###### Train and speak to HCN_go_Dummy agent
#AGENT_CONFIG_PATH = '../agent_configs/hcn_go_dummy.json'
#train_agent_models(AGENT_CONFIG_PATH)
#interact_agent(AGENT_CONFIG_PATH)

##### Train and speak to HCN_go skill separately
MODEL_CONFIG_PATH = 'config.json'
train_model_from_config(MODEL_CONFIG_PATH)
interact_model(MODEL_CONFIG_PATH)
