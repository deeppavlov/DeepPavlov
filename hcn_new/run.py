#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deeppavlov.training.train import train_agent_models, train_model_from_config
from deeppavlov.infering.infer import interact_agent, interact_model

from dstc2_dataset_reader import DSTC2DatasetReader
from models.hcn import HybridCodeNetworkBot
from commutators.random_commutator import RandomCommutator
from dummy_skill.dummy import DummySkill

###### Train and speak to HCN_go_Dummy agent
#AGENT_CONFIG_PATH = '../agent_configs/hcn_go_dummy.json'
#train_agent_models(AGENT_CONFIG_PATH)
#interact_agent(AGENT_CONFIG_PATH)

##### Train and speak to HCN_go skill separately
MODEL_CONFIG_PATH = 'config.json'
train_model_from_config(MODEL_CONFIG_PATH)
interact_model(MODEL_CONFIG_PATH)
