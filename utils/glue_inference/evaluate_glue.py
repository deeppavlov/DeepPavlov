from lib2to3.pgen2.token import BACKQUOTE

from zmq import PROTOCOL_ERROR_ZAP_BAD_REQUEST_ID
import _pickle as cPickle
import numpy as np
import sys
import os
from tqdm import tqdm
from deeppavlov import build_model
import json
from collections import defaultdict
from copy import deepcopy
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import pandas as pd
from deeppavlov.dataset_readers.huggingface_dataset_reader import preprocess_record, add_num_examples
import sys
from sklearn.metrics import confusion_matrix
from deeppavlov.dataset_readers.huggingface_dataset_reader import preprocess_wsc
from deeppavlov.dataset_readers.huggingface_dataset_reader import preprocess_multirc
from deeppavlov import build_model
from deeppavlov.models.preprocessors.torch_transformers_preprocessor import TorchTransformersPreprocessor
from torch.distributions import Categorical
from scipy.stats import pearsonr
import torch

if not os.path.exists(predicts):
    os.mkdir(predicts)


def accuracy(a, b):
    data = defaultdict(lambda: defaultdict(int))
    true = 0
    tot = 0
    for ii, jj in zip(a, b):
	data[ii][jj] += 1
	if ii == jj:
	    true += 1
	tot += 1
    if true == 0:
	breakpoint()
    print(data)
    return true / tot


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
	if isinstance(obj, np.integer):
	    return int(obj)
	if isinstance(obj, np.floating):
	    return float(obj)
	if isinstance(obj, np.ndarray):
	    return obj.tolist()
	return super(NpEncoder, self).default(obj)


ids = None

class Task:
    def __init__(self, name, filename, classes, task_id):
	self.name = name
	self.filename = filename
	self.classes = classes
	self.task_id = task_id




tasks = [Task('cola', 'CoLA.tsv', ['not_entailment', 'entailment'], 0),
	     Task('sst2', 'SST-2.tsv', [0, 1], 1),
	     Task('qqp', 'QQP.tsv', [0, 1], 2),
	     Task('mrpc', 'MRPC.tsv',[0, 1], 3),
	     Task('rte', 'RTE.tsv',  ['entailment', 'not_entailment'], 4),
	     Task('mnli-m', 'MNLI-m.tsv',  ['entailment', 'neutral', 'contradiction'], 5),
	     Task('mnli-mm', 'MNLI-mm.tsv',  ['entailment', 'neutral', 'contradiction'], 5),
	     Task('qnli', 'QNLI.tsv', ['entailment', 'not_entailment'], 6),
	     Task('stsb', 'STS-B.tsv', d [1], 7),
	     Task('ax', 'AX.tsv', ['entailment', 'not_entailment'], 4)]
tasks_to_train = [Task('cola', 'CoLA.tsv', default_sst_batch, ['not_entailment', 'entailment'], 0),
	              Task('sst2', 'SST-2.tsv', default_sst_batch, [0, 1], 1),
	              Task('qqp', 'QQP.tsv', default_rte_batch, [0, 1], 2),
	              Task('mrpc', 'MRPC.tsv', default_rte_batch, [0, 1], 3),
	              Task('rte', 'RTE.tsv', default_rte_batch, ['entailment', 'not_entailment'], 4),
	              Task('mnli', 'MNLI-m.tsv', default_rte_batch, ['entailment', 'neutral', 'contradiction'], 5),
	              Task('qnli', 'QNLI.tsv', default_rte_batch, ['entailment', 'not_entailment'], 6),
	              Task('stsb', 'STS-B.tsv', default_rte_batch, [1], 7)]


def get_glue_metric(task,test_mode='test', mode=None, MAX_LEN=1000000, log_dict=True):
    look_name = task.name.split('-m')[0]
    split = test_mode
    if task.name == 'ax':
        split = 'test'
    elif 'mnli' not in task.name:
        name = task.name
    elif task.name == 'mnli-m':
        name = 'mnli_matched'
        split = split + '_matched'
    elif task.name == 'mnli-mm':
        name = 'mnli_mismatched'
        split = split + '_mismatched'
    default_batch = [j.default_batch for j in tasks]
    try:
        dataset = load_dataset("glue", look_name, split=split)
    except Exception as e:
        print(e)
        breakpoint()
        dataset = None
        assert False
    
    loader = DataLoader(dataset, batch_size=1)
    predictions = []
    labels = []
    for batch in tqdm(loader):
        if task.name in ['cola','sst2']:
            examples = batch['sentence']
        elif task.name in ['rte', 'axb', 'stsb', 'mrpc']:
            examples = [(sentence1, sentence2) for sentence1, sentence2 in zip(batch['sentence1'], batch['sentence2'])]
        elif task.name == 'qqp':
            examples = [(sentence1, sentence2) for sentence1, sentence2 in zip(batch['question1'], batch['question2'])] 
        elif task.name == 'qnli':
            examples = [(sentence1, sentence2) for sentence1, sentence2 in zip(batch['question'], batch['sentence'])] 
        elif task.name in ['mnli-m', 'mnli-mm','mnli', 'ax']:
            examples = [(sentence1, sentence2) for sentence1, sentence2 in zip(batch['hypothesis'], batch['premise'])]
        else:
            raise Exception(F'Unsupported taskname {task.name}')
                                                    
        batch_to_use = [[] for _ in tasks_to_train] + [[None] for _ in tasks_to_train]
        batch_to_use[task.task_id] = input_to_features(examples)
        try:
            breakpoint()
            preds = model[model_index](*batch_to_use)[task.task_id][0]
            if task.name == 'stsb':
                new_pred = min(5, max(0,preds[0]))
                predictions = predictions + [new_pred]
            else:
                new_pred = [np.argmax(preds)]
                if test_mode == 'test' and 'mnli' in task.name:
                    classes =['entailment', 'contradiction', 'neutral']
                else:
                    classes = task.classes
                predictions = predictions + [classes[int(k)] for k in new_pred]

        except Exception as e:
            print(e)
            breakpoint()
        default_pred = pd.DataFrame({'predictions': predictions}).to_csv(f'{submit_dir}/{task.filename}', sep='\t')
            

def obtain_glue_predicts(tasks, test_mode='test', mode=None, MAX_LEN=1000000, log_dict=True):
    global model, CONFIG, old_args
    submit_dir = 'predicts/'+CONFIG.split('.json')[0]
    if not os.path.exists(submit_dir):
        os.mkdir(submit_dir)
    accuracies = dict()
    for task in tasks[::-1]:  # dictionary:
        answer = get_glue_metric(task,test_mode=test_mode,MAX_LEN=MAX_LEN,log_dict=log_dict, submit_dir=submit_dir)
    return accuracies



np.random.seed(1488666)

csv_file_dict = {'train': 'train.tsv', 'validation': 'val.tsv', 'test': 'test.tsv'}

CONFIG = f'{sys.argv[1]}'

if CONFIG == 'onetask':
   CONFIGS = ['config_'+k+'.json' for k in ['mrpc','rte','stsb','cola','sst2','qnli','qqp','mnli']]
elif CONFIG == 'multitask':
   CONFIGS = ['config_glue.json']
elif CONFIG == 'all':
   CONFIGS = ['config_'+k+'.json' for k in ['mrpc','rte','stsb','cola','sst2','qnli','qqp','mnli', 'all']]
for CONFIG in CONFIGS:
    print(f'Eval for {CONFIG}')
    TASK_NAME = CONFIG.split('.json')[0]
    model = build_model(CONFIG, download=False)
    if TASK_NAME in [k.name for k in tasks_to_train]:
        print(f'Assuming one_task')
        tasks_to_train=[task for task in tasks if task.name == TASK_NAME]
        assert len(tasks_to_train) == 1, breakpoint()
        tasks_to_train[0].task_id = 0
        if task.name == 'mnli':
            tasks = [Task('mnli-m', 'MNLI-m.tsv',  ['entailment', 'neutral', 'contradiction'], 0),
		     Task('mnli-mm', 'MNLI-mm.tsv',  ['entailment', 'neutral', 'contradiction'], 0)]
        else:
            tasks = tasks_to_train

    obtain_glue_predicts(task)


