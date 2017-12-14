#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path

from deeppavlov.core.commands.train import train_agent_models, train_model_from_config
from deeppavlov.core.commands.infer import interact_agent, interact_model

from deeppavlov.skills.ner.slotfill import DstcSlotFillingNetwork
from deeppavlov.skills.ner.model_loader import load_ner_dstc_model

from deeppavlov.core.data.dataset_readers.dstc2_dataset_reader import DSTC2DatasetReader

from dstc2_dataset import DSTC2Dataset
from models.hcn import HybridCodeNetworkBot

###### Train and speak to HCN_go_Dummy agent
#AGENT_CONFIG_PATH = '../agent_configs/hcn_go_dummy.json'
#train_agent_models(AGENT_CONFIG_PATH)
#interact_agent(AGENT_CONFIG_PATH)

MODEL_CONFIG_PATH = 'config.json'

# Download pretrained ner model
config = json.load(open(MODEL_CONFIG_PATH, 'rt'))
ner_model_path = Path(config['model']['slot_filler']['model_filepath'])
if not Path(ner_model_path).exists():
    load_ner_dstc_model(ner_model_path.parent)

##### Train and speak to HCN_go skill separately
train_model_from_config(MODEL_CONFIG_PATH)
interact_model(MODEL_CONFIG_PATH)
