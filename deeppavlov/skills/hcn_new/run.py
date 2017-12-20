#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path

from deeppavlov.core.commands.train import train_agent_models, train_model_from_config
from deeppavlov.core.commands.infer import interact_agent, interact_model
from deeppavlov.core.commands.utils import set_usr_dir

MODEL_CONFIG_PATH = 'config.json'
set_usr_dir(MODEL_CONFIG_PATH)

from deeppavlov.datasets.dstc2_datasets import DSTC2DialogDataset
from deeppavlov.dataset_readers.dstc2_dataset_reader import DSTC2DatasetReader
from deeppavlov.models.ner.slotfill import DstcSlotFillingNetwork
from deeppavlov.models.ner.model_loader import load_ner_dstc_model

from src.tracker import FeaturizedTracker

from src.hcn import HybridCodeNetworkBot

###### Train and speak to HCN_go_Dummy agent
#AGENT_CONFIG_PATH = '../agent_configs/hcn_go_dummy.json'
#train_agent_models(AGENT_CONFIG_PATH)
#interact_agent(AGENT_CONFIG_PATH)


# Download pretrained ner model
#config = json.load(open(MODEL_CONFIG_PATH, 'rt'))
#ner_model_path = Path(config['model']['slot_filler']['model_filepath'])
#if not Path(ner_model_path).exists():
#    load_ner_dstc_model(ner_model_path.parent)

##### Train and speak to HCN_go skill separately
train_model_from_config(MODEL_CONFIG_PATH)
interact_model(MODEL_CONFIG_PATH)
