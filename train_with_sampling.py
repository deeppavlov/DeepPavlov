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

np.random.seed(1488666)

csv_file_dict = {'train': 'train.tsv', 'validation': 'val.tsv', 'test': 'test.tsv'}

SAMPLING = f'{sys.argv[1]}'
try:
    MODE = f'{sys.argv[2]}'
except:
    MODE = 'englue'
print(f'Run with sampling {SAMPLING} mode {MODE}')
os.chdir('/cephfs/home/karpov.d/DeepPavlov/sampling')
CONFIG_DICT = {'rusuperglue': {'plain': 'rusuperglue_plain.json',
                               'plain_dsg': 'rusuperglue_plain_dsg.json',
                               'anneal': 'rusuperglue_anneal.json',
                               'anneal_dsg': 'rusuperglue_anneal_dsg.json'
                               },
               'ensuperglue': {'plain': 'ensuperglue_plain.json',
                               'uniform': 'ensuperglue_uniform.json',
                               'anneal': 'ensuperglue_anneal.json',
                               'uncertain_test':'ensuperglue_uncertain_test.json'
                               },
               'englue': {'plain': 'englue_plain.json',
                          'plain_dsg': 'englue_plain_dsg.json',
                          'anneal': 'englue_anneal.json',
                          'anneal_dsg': 'englue_anneal_dsg.json',
                          'uniform': 'englue_uniform.json',
                          'uniform_dsg': 'englue_uniform_dsg.json',
                          'uncertain': 'englue_uncertain.json',
                          'uncertain_dsg': 'englue_uncertain_dsg.json',
                          'uncertain_test': 'englue_uncertain_test.json'
                          },
               'englue_3': {'plain': 'englue_plain_3.json',
                            'plain_dsg': 'englue_plain_dsg_3.json',
                            'anneal': 'englue_anneal_3.json',
                            'anneal_dsg': 'englue_anneal_dsg_3.json',
                            'uniform': 'englue_uniform_3.json',
                            'uniform_dsg': 'englue_uniform_dsg_3.json',
                            'uncertain': 'englue_uncertain_3.json',
                            'uncertain_dsg': 'englue_uncertain_dsg_3.json',
                            'uncertain_test': 'englue_uncertain_test_3.json'
                            },
               'englue_6': {'plain': 'englue_plain_6.json'
                            },
               'englue_2': {'plain': 'englue_plain_2.json',
                            'plain_dsg': 'englue_plain_dsg_2.json',
                            'anneal': 'englue_anneal_2.json',
                            'anneal_dsg': 'englue_anneal_dsg_2.json',
                            'uniform': 'englue_uniform_2.json',
                            'uniform_dsg': 'englue_uniform_dsg_2.json',
                            'uncertain': 'englue_uncertain_2.json',
                            'uncertain_dsg': 'englue_uncertain_dsg_2.json',
                            'uncertain_test':'englue_uncertain_test_2.json',
                            'reinforce_anneal': 'englue_reinforce_anneal_2.json',
                            'reinforce_plain': 'englue_reinforce_plain_2.json',
                            'reinforce_uniform': 'englue_reinforce_uniform_2.json',
                            'reinforce_uncertain': 'englue_uncertain_plain_2.json'
                            }}

CONFIG = CONFIG_DICT[MODE][SAMPLING]
model = build_model(CONFIG, download=False)

if not os.path.exists(SAMPLING):
    os.mkdir(SAMPLING)

if 'rusuperglue' in MODE:
    raise Exception('no pickled dataset - please obtain the one')
elif 'ensuperglue' in MODE:
    PICKLE_FILE = 'ensuperglue.pkl'
elif 'englue' in MODE:
    PICKLE_FILE = 'glue1.pkl'
else:
    raise Exception(f'Unsupported mode {MODE}')

data = cPickle.load(open(PICKLE_FILE, 'rb'))
types_by_task = {'englue_3': ['mrpc', 'rte', 'cola'], 'englue_2': ['mrpc', 'rte'],
                 'englue_6':['cola','sst2','mrpc','rte','qnli','stsb']}
if MODE in types_by_task:  # filter
    for key in data:
        data[key] = {task: data[key][task] for task in data[key] if task in types_by_task[MODE]}


use_samples_by_task = {task: 1 for task in data['train']}
accuracies_by_task = {task: [] for task in data['train']}


stop_mode_tasks = set()

BATCH_SIZE = 32
if '_test' in SAMPLING:
    BATCH_SIZE = 32
elif 'superglue' in MODE:
    BATCH_SIZE = 8
NUM_TRAIN_EPOCHS = 5
DSG_STEPS = 4
LR_DROP_PATIENCE = 2
LR_DIVISOR = 2
EARLY_STOP_PATIENCE = 5
MAX_VALID_SAMPLES_BY_TASK = 1500000
# print({s: len(data['train'][s]) for s in data['train']})
# breakpoint()
STEPS_PER_EPOCH = sum([len(data['train'][s]) for s in data['train']]) // BATCH_SIZE
# STEPS_PER_EPOCH = 20
NUM_TASKS = len(data['train'])
MODEL_NAME = 'deeppavlov.models.multitask_pal_bert.multitask_pal_bert.MultiTaskPalBert'
SCORE_TIMES_PER_EPOCH = 2 # 2 times per epochs we score

accuracies = []

POLICY = None


class SamplingIds:
    def _make_ids(self,task_name):
        ids = [(i,min(i+self.bin_size,len(data['train'][task_name])))
               for i in range(0,len(data['train'][task_name]),self.bin_size)]
        np.random.shuffle(ids)
        self._ids[task_name] = set(ids)
    def __init__(self,bin_size=BATCH_SIZE):
        global data
        self._ids=dict()
        self.bin_size=bin_size
        for task_name in data['train']:
            self._make_ids(task_name)

    def sample(self, task_name):
        if not len(self._ids[task_name]):
            self._make_ids(task_name)
        (start_id, end_id) = self._ids[task_name].pop()
        return (start_id, end_id)
IdsForNormalSampling = SamplingIds()


class IdsForUncertaintySampling:
    def __init__(self):
        self.ids = dict()
        self.reload()

    def reload(self, key=None):
        global data
        if key is None:
            names = [k for k in data['train']]
        else:
            names = [key]
        for name in names:
            self.ids[name] = set([k for k in range(len(data['train'][name]))])

    def sample(self, name, batch_size):
        ids_to_sample = np.random.choice(list(self.ids[name]),
                                         size=min(batch_size, len(self.ids[name])),
                                         replace=False)
        self.ids[name] = self.ids[name] - set(ids_to_sample.tolist())
        if len(self.ids[name]) == 0:
            self.reload(key=name)
        return ids_to_sample


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


def process_superglue():
    # RTE is already there
    for dir1, dir2 in zip(['/cephfs/home/karpov.d/Data/en_superglue/CB/',
                           '/cephfs/home/karpov.d/Data/ru_superglue/RCB/',
                           '/cephfs/home/karpov.d/Data/ru_superglue/TERRa/',
                           '/cephfs/home/karpov.d/Data/ru_superglue/RUSSE/',
                           '/cephfs/home/karpov.d/Data/en_superglue/WiC/',
                           '/cephfs/home/karpov.d/Data/en_superglue/AX-b/',
                           '/cephfs/home/karpov.d/Data/ru_superglue/LiDiRus/',
                           '/cephfs/home/karpov.d/Data/ru_superglue/DaNetQA/',
                           '/cephfs/home/karpov.d/Data/en_superglue/BoolQ/',
                           '/cephfs/home/karpov.d/Data/en_superglue/WSC/'],
                          ['/cephfs/home/karpov.d/GLUE/Data/CB/',
                           '/cephfs/home/karpov.d/GLUE/Data/RCB/',
                           '/cephfs/home/karpov.d/GLUE/Data/TERRa/',
                           '/cephfs/home/karpov.d/GLUE/Data/RUSSE/',
                           '/cephfs/home/karpov.d/GLUE/Data/WiC/',
                           '/cephfs/home/karpov.d/GLUE/Data/AX-b/',
                           '/cephfs/home/karpov.d/GLUE/Data/LiDiRus/',
                           '/cephfs/home/karpov.d/GLUE/Data/DaNetQA/',
                           '/cephfs/home/karpov.d/GLUE/Data/BoolQ/']):
        print(dir1)
        for file in [k for k in os.listdir(dir1) if '.jsonl' in k]:
            if not os.path.exists(dir2):
                os.mkdir(dir2)
            print(file)
            if 'dev' in file:
                raise Exception(file)
            new_file = open(dir2 + file.replace('.jsonl', '.tsv').replace('val', 'validation'), 'w')
            new_file.write('index\tsentence1\tsentence2\tlabel\n')
            for line in open(dir1 + file, 'r').readlines():
                k = json.loads(line.strip().replace('\\n', ''))
                assert str(k).count('\n') == 0
                if 'sentence1' in k:
                    k['premise'] = k['sentence1']
                    k['hypothesis'] = k['sentence2']
                elif 'question' in k:
                    k['premise'] = k['question']
                    k['hypothesis'] = k['passage']
                elif 'premise' not in k:
                    raise Exception(k)
                if 'label' in k:
                    assert 'test' not in file
                    new_file.write('\t'.join([str(s) for s in [k['idx'], k['premise'],
                                                               k['hypothesis'], str(k['label']).lower()]]) + '\n')
                else:
                    assert 'test' in file
                    new_file.write('\t'.join([str(s) for s in [k['idx'], k['premise'], k['hypothesis']]]) + '\n')
                    # raise Exception(k)
            # print(file)
            # print('*')
            # print(dir2+file.replace('.jsonl', '.tsv'))
            new_file.close()


class Task:
    def __init__(self, name, filename, default_batch, classes, task_id):
        self.name = name
        self.filename = filename
        self.default_batch = default_batch
        self.classes = classes
        self.task_id = task_id


default_sst_batch = [(-1, 'a')]
default_rte_batch = [(-1, ('a', 'a'))]
default_copa_batch = [(-1, ('a', ['a', 'a']))]
default_record_batch = [(-1, (1, 'a', 'a', 'a', 444))]
if 'rusuperglue' in MODE:
    print('RUSUPERGLUE task set')
    tasks = [Task('rwsd', 'RWSD.jsonl', default_rte_batch, ["False", "True"], 0),
             Task('muserc', 'MuSeRC.jsonl', default_rte_batch, [0, 1], 1),
             Task('rcb', 'RCB.jsonl', default_rte_batch, ['contradiction', 'neutral', 'entailment'], 2),
             Task('rucos', 'RuCoS.jsonl', default_record_batch, [1, 0], 3),
             Task('danetqa', 'DaNetQA.jsonl', default_rte_batch, ['false', 'true'], 4),
             Task('parus', 'PARus.jsonl', default_copa_batch, [0, 1], 5),
             Task('terra', 'TERRa.jsonl', default_rte_batch, ['entailment', 'not_entailment'], 6),
             Task('russe', 'RUSSE.jsonl', default_rte_batch, ['false', 'true'], 7),
             Task('lidirus', 'LiDiRus.jsonl', default_rte_batch, ['not_entailment', 'entailment'], 6)]
    tasks_to_train = tasks[:-1]
elif 'ensuperglue' in MODE:
    print('ENSUPERGLUE task set')
    tasks = [Task('wsc', 'WSC.jsonl', default_rte_batch, ["False", "True"], 0),
             Task('multirc', 'MultiRC.jsonl', default_rte_batch, [0, 1], 1),
             Task('cb', 'CB.jsonl', default_rte_batch, ['contradiction', 'entailment', 'neutral'], 2),
             Task('record', 'ReCoRD.jsonl', default_record_batch, [1, 0], 3),
             Task('boolq', 'BoolQ.jsonl', default_rte_batch, ['false', 'true'], 4),
             Task('copa', 'COPA.jsonl', default_copa_batch, [0, 1], 5),
             Task('rte', 'RTE.jsonl', default_rte_batch, ['entailment', 'not_entailment'], 6),
             Task('wic', 'WiC.jsonl', default_rte_batch, ['false', 'true'], 7),
             Task('ax-b', 'Ax-b.jsonl', default_rte_batch, ['entailment', 'not_entailment'], 6)]
    tasks_to_train = tasks[:-1]
elif MODE == 'englue':
    tasks = [Task('cola', 'CoLA.tsv', default_sst_batch, ['not_entailment', 'entailment'], 0),
             Task('sst2', 'SST-2.tsv', default_sst_batch, [0, 1], 1),
             Task('qqp', 'QQP.tsv', default_rte_batch, [0, 1], 2),
             Task('mrpc', 'MRPC.tsv', default_rte_batch, [0, 1], 3),
             Task('rte', 'RTE.tsv', default_rte_batch, ['entailment', 'not_entailment'], 4),
             Task('mnli-m', 'MNLI-m.tsv', default_rte_batch, ['entailment', 'neutral', 'contradiction'], 5),
             Task('mnli-mm', 'MNLI-mm.tsv', default_rte_batch, ['entailment', 'neutral', 'contradiction'], 5),
             Task('qnli', 'QNLI.tsv', default_rte_batch, ['entailment', 'not_entailment'], 6),
             Task('stsb', 'STS-B.tsv', default_rte_batch, [1], 7),
             Task('ax', 'AX.tsv', default_rte_batch, ['entailment', 'not_entailment'], 4)]
    tasks_to_train = [Task('cola', 'CoLA.tsv', default_sst_batch, ['not_entailment', 'entailment'], 0),
                      Task('sst2', 'SST-2.tsv', default_sst_batch, [0, 1], 1),
                      Task('qqp', 'QQP.tsv', default_rte_batch, [0, 1], 2),
                      Task('mrpc', 'MRPC.tsv', default_rte_batch, [0, 1], 3),
                      Task('rte', 'RTE.tsv', default_rte_batch, ['entailment', 'not_entailment'], 4),
                      Task('mnli', 'MNLI-m.tsv', default_rte_batch, ['entailment', 'neutral', 'contradiction'], 5),
                      Task('qnli', 'QNLI.tsv', default_rte_batch, ['entailment', 'not_entailment'], 6),
                      Task('stsb', 'STS-B.tsv', default_rte_batch, [1], 7)]
elif MODE == 'englue_6':
    tasks = [Task('cola', 'CoLA.tsv', default_sst_batch, ['not_entailment', 'entailment'], 0),
             Task('sst2', 'SST-2.tsv', default_sst_batch, [0, 1], 1),
             Task('mrpc', 'MRPC.tsv', default_rte_batch, [0, 1], 2),
             Task('rte', 'RTE.tsv', default_rte_batch, ['entailment', 'not_entailment'], 3),
             Task('qnli', 'QNLI.tsv', default_rte_batch, ['entailment', 'not_entailment'], 4),
             Task('stsb', 'STS-B.tsv', default_rte_batch, [1], 5)]
    tasks_to_train = tasks
elif MODE == 'englue_3':
    tasks = [Task('cola', 'CoLA.tsv', default_sst_batch, ['not_entailment', 'entailment'], 0),
             Task('mrpc', 'MRPC.tsv', default_rte_batch, [0,1], 1),
             Task('rte', 'RTE.tsv', default_rte_batch, ['entailment', 'not_entailment'], 2)]
    tasks_to_train = tasks
elif MODE == 'englue_2':
    tasks = [Task('mrpc', 'MRPC.tsv', default_rte_batch, [0,1], 0),
             Task('rte', 'RTE.tsv', default_rte_batch, ['entailment', 'not_entailment'], 1)]
    tasks_to_train = tasks
else:
    raise Exception(f'Unsupported mode {MODE}')


old_args = None


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
        batch_to_use[task.task_id] = (examples)
        try:
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
        if test_mode == 'validation':
            if task.name != 'stsb':
                labels = labels + [task.classes[int(k)] for k in batch['label']]
            elif task.name == 'stsb':
                labels = labels + batch['label'].tolist()
            if len(labels) > MAX_LEN:
                break
            assert len(labels) == len(predictions), breakpoint()
    if 'validation' in test_mode:
        if task.name != 'ax' :
            if task.name != 'stsb':
                metric = accuracy
            else:
                metric = lambda x, y: 100*pearsonr(r, y)
            if log_dict:
                true_to_pred = defaultdict(lambda: defaultdict(int))
                for label, prediction in zip(labels, predictions):
                    true_to_pred[label][prediction] += 1
                print(true_to_pred)
            ans = metric(predictions, labels)
            if ans == 0:
                breakpoint()
            return ans
    else:
        try:
            default_pred = pd.DataFrame({'predictions': predictions})
        except Exception as e:
            # print(e)
            breakpoint()
            raise e
        default_pred.to_csv(f'{submit_dir}/{task.filename}', sep='\t')
            

def obtain_glue_predicts(test_mode='test', mode=None, MAX_LEN=1000000, log_dict=True):
    global model, CONFIG, old_args
    submit_dir = CONFIG.split('.json')[0]
    if not os.path.exists(submit_dir):
        os.mkdir(submit_dir)
    accuracies = dict()
    for task in tasks[::-1]:  # dictionary:
        answer = get_glue_metric(task,test_mode=test_mode,MAX_LEN=MAX_LEN,log_dict=log_dict)
        if test_mode == 'validation':
            accuracies[task.name] = answer
    return accuracies


def obtain_superglue_predicts(mode='ru', test_mode='test', MAX_LEN=1000000, log_dict=False, DEBUG=False,
                              default_batch_size=1):
    global model, CONFIG
    submit_dir = CONFIG.split('.json')[0]
    if not os.path.exists(submit_dir):
        os.mkdir(submit_dir)
    accuracies = {}
    if mode == 'ru':
        dataset_name = 'russian_super_glue'
    elif mode == 'en':
        counter = None
        dataset_name = 'super_glue'
    default_batch = [x.default_batch for x in tasks]

    def get_prediction(classes, i, batch_i, return_probas=False):
        try:
            batch = default_batch.copy()
            batch[i] = batch_i
            if not DEBUG:
                args = batch + [[None] for _ in range(len(batch))]
                breakpoint()
                predictions = model(*args)
                predicted_probas = predictions[i][0]
                if return_probas:
                    return predicted_probas
                predicted_class = np.argmax(predicted_probas)
                prediction = classes[predicted_class]
                return prediction
            elif DEBUG:
                if return_probas:
                    return [0.4, 0.6]
                return 0
        except Exception as e:
            print(classes)
            print(i)
            print(e)
            raise e
    for task in tasks_to_train:
            counter = defaultdict(int)
            true_count = 0
            total_count = 0
            true_to_total = defaultdict(lambda: defaultdict(int))
            print(task.name)
            i = task.task_id
            label_counter = defaultdict(int)
            filename = f'{submit_dir}/{task.filename}'
            output = []
            if task.name in ['terra', 'rte', 'rcb', 'cb', 'russe', 'wic', 'danetqa', 'boolq', 'lidirus',
                        'axb']:  # was only rcb and cb
                label_counter = defaultdict(int)
                data_name = filename.split('/')[-1].split('.')[0]
                toload_dir = f'/cephfs/home/karpov.d/GLUE/Data/{data_name}'
                if not os.path.exists(toload_dir):
                    os.mkdir(toload_dir)
                loader = open(f'{toload_dir}/{csv_file_dict[test_mode]}', 'r').readlines()
                if test_mode != 'test' and len(loader) > MAX_LEN:
                    loader = loader[:MAX_LEN]
                # writer=open(f'/cephfs/home/karpov.d/GLUE/Data/{dataset_name}/new_{csv_file_dict[MODE]}','w')
                for batch in tqdm(loader):
                    line = deepcopy(batch)
                    names = batch.strip().split('\t')
                    batch = {ln: name for ln, name in zip(['idx', 'hypothesis', 'premise', 'label'], names)}
                    index = batch['idx']
                    if index != 'index':
                        # print(index)
                        hypothesis = batch["hypothesis"]
                        premise = batch["premise"]
                        if task in ['lidirus', 'ax-b']:
                            j = [ii for ii in range(len(tasks)) if tasks[ii].name in ['terra', 'rte']][0]
                        else:
                            j = i
                        TASK_BATCH = [(j, (hypothesis, premise))]
                        prediction = get_prediction(task.classes, j, TASK_BATCH)
                        if test_mode != 'test':
                            true_to_total[prediction][batch['label']] += 1
                            if batch['label'] == prediction:
                                true_count += 1
                            total_count += 1
                        if total_count == MAX_LEN:
                            break
                        output.append(dict(idx=int(index), label=prediction))
                cc = 0
                if test_mode == 'test':
                    with open(filename, "w") as file:
                        for element in output:
                            file.write(json.dumps(element, ensure_ascii=False, cls=NpEncoder) + "\n")
                            cc += 1
            elif task.name in ['parus', 'copa']:
                # from deeppavlov.dataset_readers.huggingface_dataset_reader import preprocess_copa
                # dataset = dataset.map(preprocess_copa, batched=True, fn_kwargs={"lang": "ru"})
                dataset = load_dataset(dataset_name, task.name, split=test_mode)
                loader = DataLoader(dataset, batch_size=1)

                if task == 'parus':
                    question_dict = {
                        "cause": "Что было причиной этого?",
                        "effect": "Что случилось в результате?",
                    }
                else:
                    question_dict = {
                        "cause": "What was the cause of this?",
                        "effect": "What happened as a result?"}
                output = []
                for batch in tqdm(loader):
                    question = question_dict[batch["question"][0]]
                    context = f"{batch['premise'][0]} {question}"
                    index = batch["idx"]
                    choices = [batch["choice1"][0], batch["choice2"][0]]
                    TASK_BATCH = [(i, (context, choices))]
                    pred = get_prediction(task.classes, i, TASK_BATCH)
                    label = pred
                    if test_mode != 'test':
                        true_class = int(batch['label'][0])
                        true_to_total[true_class][label] += 1
                        if true_class == label:
                            true_count += 1
                        total_count += 1
                        if total_count == MAX_LEN:
                            break
                    label_counter[label] += 1
                    output.append(dict(idx=int(index), label=label))
                k = 0
                with open(filename, "w") as file:
                    for element in output:
                        file.write(json.dumps(element, ensure_ascii=False, cls=NpEncoder) + "\n")
                        k += 1
            elif task.name in ['rwsd', 'wsc']:
                # old type
                dataset = load_dataset(dataset_name, task.name, split=test_mode)
                dataset = dataset.map(
                    preprocess_wsc,
                    batched=True,
                    remove_columns=["span1_index", "span2_index", "span1_text", "span2_text"],
                )
                loader = DataLoader(dataset, batch_size=1)
                output = []
                for batch in tqdm(loader):
                    index = batch["idx"][0]
                    context = batch["text"][0]
                    answer = batch["answer"][0]
                    TASK_BATCH = [(i, (context, answer))]
                    prediction = get_prediction(task.classes, i, TASK_BATCH)
                    output.append(dict(idx=int(index), label=prediction))
                    if test_mode != 'test':
                        true_class = task.classes[int(batch['label'][0].int())]
                        true_to_total[true_class][prediction] += 1
                        if true_class == prediction:
                            true_count += 1
                        total_count += 1
                        if total_count == MAX_LEN:
                            break
                k = 0
                with open(filename, "w") as file:
                    for element in output:
                        file.write(json.dumps(element, ensure_ascii=False, cls=NpEncoder) + "\n")
                        k += 1
            elif task.name in ['muserc', 'multirc']:
                dataset = load_dataset(dataset_name, task.name, split=test_mode)
                dataset = dataset.map(
                    preprocess_multirc, batched=True, remove_columns=["paragraph", "question"]
                )
                loader = DataLoader(dataset, batch_size=1)
                output = {}
                for batch in tqdm(loader):
                    # raise Exception(batch)
                    indices = batch["idx"]
                    paragraph_idx = int(indices["paragraph"][0])
                    question_idx = int(indices["question"][0])
                    answer_idx = int(indices["answer"][0])
                    context = batch["context"][0]
                    answer = batch["answer"][0]
                    TASK_BATCH = [(i, (context, answer))]
                    label = get_prediction(task.classes, i, TASK_BATCH)
                    label_counter[label] += 1
                    if test_mode != 'test':
                        true_class = task.classes[int(batch['label'][0].int())]
                        true_to_total[true_class][label] += 1
                        if true_class == label:
                            true_count += 1
                        total_count += 1
                        if total_count == MAX_LEN:
                            break
                    if paragraph_idx not in output:
                        output[paragraph_idx] = dict(
                            idx=paragraph_idx, passage=dict(
                                questions=[
                                    dict(
                                        idx=question_idx,
                                        answers=[
                                            dict(
                                                idx=answer_idx,
                                                label=label
                                            )
                                        ]
                                    )
                                ]
                            )
                        )
                    questions = output[paragraph_idx]["passage"]["questions"]
                    question_indices = set(el["idx"] for el in questions)
                    if question_idx not in question_indices:
                        output[paragraph_idx]["passage"]["questions"].append(dict(
                            idx=question_idx, answers=[dict(idx=answer_idx, label=label)]
                        ))
                    else:
                        for question in output[paragraph_idx]["passage"]["questions"]:
                            if question["idx"] == question_idx:
                                question["answers"].append(dict(idx=answer_idx, label=label))
                output = [value for _, value in output.items()]
                k = 0
                # print(true_to_total)
                # print('cnt')
                # print(label_counter)
                with open(filename, "w") as file:
                    for element in output:
                        # print(element)
                        file.write(json.dumps(element, ensure_ascii=False, cls=NpEncoder) + "\n")
                        k += 1
            elif task.name in ['rucos', 'record']:
                dataset = load_dataset(dataset_name, task.name, split=test_mode)

                dataset = dataset.map(
                    preprocess_record, batched=True, remove_columns=["answers"]
                ).map(
                    add_num_examples, batched=True, batch_size=None
                )
                loader = DataLoader(dataset, batch_size=1)
                index_list = []
                entities_list = []
                predictions = []
                true_count, total_count = 0, 0
                for batch in tqdm(loader):
                    indices = batch["idx"]
                    queries = batch["query"]
                    passages = batch["passage"]
                    entities = batch["entities"]
                    num_examples = batch["num_examples"]
                    rucos_batch = [(i, (id, query, passage, entity, n))
                                   for id, query, passage, entity, n in
                                   zip(indices, queries, passages, entities, num_examples)]
                    probas = get_prediction(task.classes, i, rucos_batch, return_probas=True)
                    prediction = probas[1]
                    index_list.extend(indices)
                    entities_list.extend(entities)
                    predictions.append(prediction)
                    # if DEBUG:
                    #    break
                    if test_mode != 'test':
                        true_class = int(batch['label'][0].int())
                        pred_round = int(round(prediction, 0))
                        true_to_total[true_class][pred_round] += 1
                        if true_class == pred_round:
                            true_count += 1
                        total_count += 1
                        if total_count == MAX_LEN:
                            break
                output = defaultdict(
                    lambda: {
                        "predicted": [],
                        "probability": []
                    }

                )
                for index, entity, prediction in zip(index_list, entities_list, predictions):
                    output[index]["predicted"].append(entity)
                    output[index]["probability"].append(float(prediction))
                submission = []
                for key, value in output.items():
                    answer_index = np.argmax(value["probability"])
                    label_counter[answer_index] += 1
                    answer = value["predicted"][answer_index]
                    submission.append(
                        {
                            "idx": int(key.split("-")[-2]),
                            "label": answer
                        })
                with open(filename, "w") as file:
                    k = 0
                    for element in sorted(submission, key=lambda d: d["idx"]):
                        file.write(json.dumps(element, ensure_ascii=False, cls=NpEncoder) + "\n")
                        k += 1
            else:
                breakpoint()
                raise ValueError
            if test_mode == 'validation':
                accuracies[task] = true_count / total_count
            if log_dict:
                print(counter)
                print('true to total')
                print(true_to_total)
    if test_mode == 'validation':
        return accuracies


if 'superglue' in MODE:
    obtain_predicts = obtain_superglue_predicts
elif 'englue' in MODE:
    obtain_predicts = obtain_glue_predicts


def sample_with_probs(probs):
    global NUM_TASKS
    args_x = [[] for _ in range(NUM_TASKS)]
    args_y = [[] for _ in range(NUM_TASKS)]
    #breakpoint()
    task_id = np.random.choice(len(probs), p=probs)
    #breakpoint()
    name = tasks_to_train[task_id].name
    start_id, end_id = IdsForNormalSampling.sample(name)
    sampled_data = data['train'][name][start_id:end_id]
    x,y=[k[0] for k in sampled_data],[k[1] for k in sampled_data]
    args_x[task_id] = x
    args_y[task_id] = y
    if task_id == 6 and y == 2:
        breakpoint()
        assert False
    return args_x+ args_y


def use_dsg():
    print('use dsg')
    for task in tasks_to_train:
        best_accuracy_task = max([accuracies[num_epoch][task.name] for num_epoch in range(len(accuracies))])
        i = model.pipe[model_index][-1].task_names.index(task.name)
        if best_accuracy_task - accuracies[-1][task.name] > 0.5 * 1e-2 and task.name in stop_mode_tasks:  # 0.5%
            print(f'Task {i} diverged. Shift from stop mode to dsg mode')
            model.pipe[model_index][-1].gradient_accumulation_steps[i] = 1
        elif len(accuracies) > 2 and accuracies[-1][task] - accuracies[-3][task] < 0.1 * 1e-2:
            print(f'Task {i} converged. Shift from dsg mode to stop mode')
            model.pipe[model_index][-1].gradient_accumulation_steps[i] = 4
            stop_mode_tasks.add(task.name)


def sample_plain(epochs_done=0, return_probs=False):
    # Равномерно сэмплируем
    global data
    sizes = [len(data['train'][s]) for s in data['train']]
    tot = sum(sizes)
    probs = [p / tot for p in sizes]
    if return_probs:
        return probs
    return sample_with_probs(probs)


def sample_uniform(epochs_done=0, return_probs=False):
    # Равномерно сэмплируем
    probs = [1.0 / NUM_TASKS for _ in range(NUM_TASKS)]
    if return_probs:
        return probs
    return sample_with_probs(probs)


def sample_reinforce(BATCH_SIZE, eps=0.9, epochs_done=0, default_mode='anneal'):
    ##### IDEA. Learn to choose samples so as to minimise uncertainty
    print('NOT YET SUPPORTED!')
    assert False
    args = [[] for _ in range(2 * NUM_TASKS)]
    for example in range(BATCH_SIZE):
        p = np.random.uniform(0, 1)
        function_dict = {'anneal': sample_anneal, 'plain': sample_plain, 'uniform': sample_uniform,
                         'uncertain': sample_uncertainty}
        if p < eps:
            returned_args = function_dict[default_mode](BATCH_SIZE=1)
            for i in range(len(args)):
                args[i] = args[i] + returned_args[i]

    return args


def sample_anneal( epochs_done=0, return_probs=False):
    # annealed sampling
    global data
    sizes = [len(data['train'][s]) for s in data['train']]
    alpha = 1.0 - 0.8 * (epochs_done / NUM_TRAIN_EPOCHS)
    # print('Sizes ')
    # print(self._get_data_size(data))
    probs = [p ** alpha for p in sizes]
    tot = sum(probs)
    probs = [p / tot for p in probs]
    if return_probs:
        return probs
    return sample_with_probs(probs)


def get_entropy(sample, task_id):
    # breakpoint()
    try:
        global old_args, tasks
        x_args = [[] for task in tasks]
        y_args = [[] for _ in tasks]
        x_args[task_id] = input_to_features([sample])
        y_args[task_id] = [None]
        args1 = x_args + y_args
        # print('Args for get_entropy')
        # print(args1)
        assert len(args1[task_id]) > 0
        ans = entropy_from_features(args1, task_id)
        return ans
    except Exception as e:
        breakpoint()


def entropy_from_features(args1, task_id):
    try:
        predicted_probas = torch.Tensor(model.pipe[model_index][-1](*(args1))[task_id])
    except Exception as e:
        print(e)
        breakpoint()
    #print('Probas')
    #print(predicted_probas)
    entropy = float(Categorical(probs=predicted_probas).entropy())
    return entropy    


def get_samples_entropy(samples,task_id):
    import time
    t=time.time()
    x_args = [[] for task in tasks_to_train]
    y_args = [[] for _ in tasks_to_train]
    x_args[task_id] = input_to_features(samples)
    #breakpoint()
    predicted_probas = torch.Tensor(model.pipe[model_index][-1](*(x_args + y_args))[task_id][0])
    #breakpoint()
    return Categorical(predicted_probas).entropy()


def sample_uncertainty(BATCH_SIZE, log=True,old_mode=True):
    import time
    t = time.time()
    # uncertainty sampling - from ca-mtl
    global ids
    if ids is None:
        ids = IdsForUncertaintySampling()
    #breakpoint()
    entropy_list = torch.Tensor([0. for _ in range(NUM_TASKS*BATCH_SIZE)]).cuda()
    mean_task_entropies = torch.Tensor([0. for _ in range(NUM_TASKS*BATCH_SIZE)]).cuda()
    task_id_list = torch.Tensor([0 for _ in range(NUM_TASKS*BATCH_SIZE)]).to(int).cuda()
    sample_list = torch.Tensor([0 for _ in range(NUM_TASKS*BATCH_SIZE)]).to(int).cuda()
    max_mean_batch_entropy = None
    mean_entropies_from_task = []
    start_ind = 0
    names = dict()
    for task in tasks_to_train:
        names[task.task_id] = task.name
        ids_to_sample = ids.sample(name=task.name, batch_size=BATCH_SIZE)
        task_id_list[start_ind:start_ind+len(ids_to_sample)].fill_(task.task_id)
        samples_from_task = [data['train'][task.name][id_][0] for id_ in ids_to_sample]
        if task.name != 'stsb':
            entropies_from_task = get_samples_entropy(samples_from_task, task.task_id)
        else:
            mrpc_task = [task for task in tasks_to_train if task.name=='mrpc'][0]
            entropies_from_task = get_samples_entropy(samples_from_task, mrpc_task.task_id)
        entropy_list[start_ind:start_ind + len(ids_to_sample)] = entropies_from_task
        mean_task_entropies[start_ind:start_ind + len(ids_to_sample)].fill_(entropies_from_task.mean())
        start_ind+=len(ids_to_sample)
        #breakpoint()
    t=time.time()
    max_mean_batch_entropy = mean_task_entropies.max()
    entropy_list = torch.mul(entropy_list, mean_task_entropies) / max_mean_batch_entropy
    values, indices = torch.topk(entropy_list,BATCH_SIZE)
    x_args = [[] for _ in data['train']]
    y_args = [[] for _ in data['train']]
    for index in indices:
        task_id = task_id_list[index]
        task_name = names[int(task_id)]
        sample_id = sample_list[index]
        sample = data['train'][task_name][sample_id]
        x_args[task_id].append(sample[0])
        y_args[task_id].append(sample[1])
    return x_args + y_args




def sample_uncertainty_old(BATCH_SIZE, log=True,old_mode=True):
    # uncertainty sampling - from ca-mtl
    global ids
    if ids is None:
        ids = IdsForUncertaintySampling()
    if '_test' in SAMPLING:
        old_mode = False
    entropy_list = []
    task_id_list = []
    sample_list = []
    max_mean_batch_entropy = None
    mean_entropies_from_task = []
    for task in tasks_to_train:
        ids_to_sample = ids.sample(name=task.name, batch_size=BATCH_SIZE)
        samples_from_task = [data['train'][task.name][id_] for id_ in ids_to_sample]
        if task.name != 'stsb':
            entropies_from_task = [get_entropy(sample[0], task.task_id) for sample in samples_from_task]
        else:
            print('As in CA-MTL, treating sts-b example as MRPC example')
            mrpc_task = [task for task in tasks_to_train if task.name=='mrpc'][0]
            entropies_from_task = [get_entropy(sample[0], mrpc_task.task_id) for sample in samples_from_task]
        if log:
            print('raw entropies')
            print(entropies_from_task)
        # avg_task_entropy = sum(entropies_from_task)/len(entropies_from_task)
        # max_task_entropy = float(Categorical(probs=torch.Tensor([1 / len(task.classes) for _ in range(len(task.classes))])).entropy())
        # entropies_from_task = [k * avg_task_entropy / max_task_entropy for k in entropies_from_task]
        if None in entropies_from_task:
            breakpoint()
        if old_mode:
            entropies_from_task = [k / max(entropies_from_task) for k in entropies_from_task]
        else:
            batch_entropy_mean = sum(entropies_from_task) / len(entropies_from_task)
            if max_mean_batch_entropy is None or batch_entropy_mean > max_mean_batch_entropy:
                max_mean_batch_entropy = batch_entropy_mean
                mean_entropies_from_task = mean_entropies_from_task + [batch_entropy_mean for _ in entropies_from_task]
        if log:
            print('entropies')
            print(entropies_from_task)
        sample_list = sample_list + samples_from_task
        entropy_list = entropy_list + entropies_from_task
        task_id_list = task_id_list + [task.task_id for _ in range(len(samples_from_task))]
        assert len(entropies_from_task) == len(samples_from_task), breakpoint()

    if not old_mode:
        entropy_list = [entropy*mean_entropy/max_mean_batch_entropy for entropy,mean_entropy in zip(entropy_list, mean_entropies_from_task)]
        if log:
            print('all entropies')
            print(entropy_list)
    selected_indices = np.argsort(entropy_list)
    selected_indices = selected_indices[-BATCH_SIZE:]
    #breakpoint()
    if log:
        print('max entropy indices task ids')
        print(selected_indices)
        print([task_id_list[k] for k in selected_indices])
    x_args = [[] for _ in data['train']]
    y_args = [[] for _ in data['train']]
    for index in selected_indices:
        task_id, sample = task_id_list[index], sample_list[index]
        x_args[task_id].append(sample[0])
        y_args[task_id].append(sample[1])
    if log:
        print([len(k) for k in x_args])
    # if len(x_args[0]) == 0 or len(x_args[1]) == 0 and not old_mode:
    #    breakpoint()
    return x_args + y_args


def this_is_the_best(accuracies):
    if len(accuracies) == 1:
        return True
    avg_accs = [sum(j.values()) / len(j) for j in accuracies]
    if len(avg_accs) == 1:
        print(f'Initial average acc {avg_accs[-1]}')
        return True
    if avg_accs[-1] == max(avg_accs):
        print(f'Avg accs improved to {round(avg_accs[-1], 4)} from {round(avg_accs[-2], 4)}')
        return True
    else:
        print(f'Avg accs {round(avg_accs[-1], 4)} is no better than max: {round(max(avg_accs), 4)}')
        return False


def accuracy_not_improves(accuracies, patience=2, log=False):
    try:
        acc_sums = [sum(z.values()) / len(z) for z in accuracies]
        max_acc = max(acc_sums)
        max_ind = len(acc_sums) - 1
        if len(accuracies) == 1:
            print('This is initial accuracy')
            return False
        if acc_sums[max_ind] == max_acc:
            print(f'Avg acc improved from {acc_sums[max_ind - 1]} to {acc_sums[max_ind]}')
            return False
        else:
            print(f'Avg acc was {acc_sums[max_ind - 1]} and became {acc_sums[max_ind]}')
        max_acc_ind = max_ind - acc_sums[::-1].index(max_acc)  # last index of acc_sums
        if max_acc_ind + patience == max_ind:
            if log:
                print(f' {max_acc_ind + patience - max_ind} steps are left for early stopping')
            print('Ran out of patience')
            return True
        else:
            if log:
                print(f' {max_acc_ind + patience - max_ind} steps are left from early stopping')
            return False
    except Exception as e:
        breakpoint()
        assert False


def write_best_accuracy_to_file(accuracies, filename):
    acc_sums = [sum(z.values()) / len(z) for z in accuracies]
    max_acc = max(acc_sums)
    index_of_max_accuracy = acc_sums.index(max_acc)
    writer = open(filename, 'w')
    writer.write('Best accs ' + str(accuracies[index_of_max_accuracy]) + '\n')
    writer.write('All accs ' + str(accuracies))
    writer.close()


# ALSO - lr drop, lr drop patience? Not now.
last_ind = 0
model_index = None
for i in range(len(model.pipe)):
    if MODEL_NAME in str(model.pipe[i][-1]):
        model_index = i
if model_index is None:
    raise Exception('No BERT model found')

preprocessor = TorchTransformersPreprocessor(vocab_file=model.pipe[model_index][-1].backbone_model,
                                             max_seq_length=model.pipe[model_index][-1].max_seq_len,
                                             do_lower_case='uncased' in model.pipe[model_index][-1].backbone_model)


def input_to_features(x):
    x_old=x
    #breakpoint()
    if len(x) == 0:
        return x
    if isinstance(x[0][0],str) and isinstance(x[0][1],list):
        x_new = []
        for value in x:
            for i in range(len(value[1])):
                x_new.append((value[0],value[1][i]))
        x = x_new
    global preprocessor
    input_texts = [[], []]
    for i in range(len(x)):
        if isinstance(x[i], str):
            input_texts[0].append(x[i])
        elif isinstance(x[i], tuple):
            input_texts[0].append(x[i][0])
            if len(x[i]) > 1:
                input_texts[1].append(x[i][1])
    if len(input_texts[1]) == 0:
        input_texts[1] = None
    if len(input_texts[0])==0:
        print('NO INPUT RECEIVED')
        print(x_old)
        breakpoint()
    features = preprocessor(texts_a=input_texts[0], texts_b=input_texts[1])
    return [features]


def transform(arguments, i):
    answer = [[] for _ in range(len(arguments))]
    if not arguments[i]:
        return None
    answer[i] = arguments[i]
    answer[i + NUM_TASKS] = arguments[i + NUM_TASKS]
    return answer


def obtain_features(debug=False):
    save_directory = f'tokenized_data'
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    for task in tasks:
        for mode_ in ['train', 'validation', 'test']:
            pass


def train(debug=False):
    print('Evaluating')
    #accuracies.append(obtain_predicts(mode='en', test_mode='validation', MAX_LEN=MAX_VALID_SAMPLES_BY_TASK, log_dict=False))
    #print(f'Accuracies {accuracies[-1]}')
    last_ind = 0
    for ep in range(NUM_TRAIN_EPOCHS):
        durations = [0 for _ in range(NUM_TASKS)]
        for step in tqdm(range(STEPS_PER_EPOCH)):
            if SAMPLING in ['anneal', 'anneal_dsg']:
                _args = sample_anneal(epochs_done=ep)
            elif SAMPLING in ['plain', 'plain_dsg']:
                _args = sample_plain()
            elif SAMPLING in ['uncertain', 'uncertain_dsg','uncertain_test']:
                _args = sample_uncertainty(BATCH_SIZE)
            elif SAMPLING in ['uniform', 'uniform_dsg']:
                _args = sample_uniform()
            elif 'reinforce' in SAMPLING:
                DEFAULT_MODE = SAMPLING.replace('_')[1]
                assert DEFAULT_MODE in ['anneal', 'plain', 'uncertain', 'uniform']
                _args = sample_reinforce(BATCH_SIZE, default_mode=DEFAULT_MODE)
            else:
                raise Exception(f'Unknown sampling mode {SAMPLING}')
            for task_index in range(NUM_TASKS):
                try:
                    durations[task_index] += len(_args[task_index])
                except Exception as e:
                    print(e)
                    breakpoint()
            losses = model.pipe[model_index][-1].train_on_batch(
                    *(_args_to_train))

            if step%1000==0:
                print(f'Epoch {ep}')
                print([len(z) for z in _args[:NUM_TASKS]])
                print(durations)
            if max(durations) / sum(durations) > 0.95 and sum(durations)>100 and False:
                breakpoint()
            if 'uncertain' in SAMPLING and False and any([len(arg_)==BATCH_SIZE for arg_ in _args]):
                print('DEBUGGING FOR UNCERTAIN SAMPLING')
                N_DEBUG_STEPS = 1000    
                breakpoint()
                for step in range(N_DEBUG_STEPS):
                    entropies=[]
                    for k in range(NUM_TASKS):
                        task_entropies = []
                        for sample in _args[k]:
                            task_entropies.append(get_entropy(sample,k))
                        if not task_entropies:
                            entropies.append(None)
                        else:
                            entropies.append(sum(task_entropies)/len(task_entropies))
                    print('entropies before train')
                    print(entropies)
                    losses = model.pipe[model_index][-1].train_on_batch(*(_args_to_train))
                    print('losses')
                    print(losses)
                    print('entropies after train')
                    for k in range(NUM_TASKS):
                        task_entropies = []
                        for sample in _args[k]:
                            task_entropies.append(get_entropy(sample,k))
                        if not task_entropies:
                            entropies.append(None)
                        else:
                            entropies.append(sum(task_entropies)/len(task_entropies))
                    print(entropies)
                    breakpoint()
        print('Epoch ended')
        duration_dict = {task.name: duration for task, duration in zip(tasks_to_train, durations)}
        print(f'Samples seen {duration_dict}')
        accuracies.append(obtain_predicts(mode='ru', test_mode='validation',
                                          MAX_LEN=MAX_VALID_SAMPLES_BY_TASK, log_dict=False))
        print(f'Accuracies {accuracies[-1]}')
        if 'dsg' in SAMPLING:
            use_dsg()
        if this_is_the_best(accuracies):
            model.pipe[model_index][-1].save()
        if accuracy_not_improves(accuracies, EARLY_STOP_PATIENCE, log=True):
            break
        if accuracy_not_improves(accuracies[last_ind:], LR_DROP_PATIENCE):
            print(f'Dividing lr by {LR_DIVISOR}')
            for param_group in model.pipe[model_index][-1].optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] / LR_DIVISOR, model.pipe[model_index][-1].min_learning_rate)
                model.pipe[model_index][-1].load()
            last_ind = len(accuracies) - 1
    print('Training finished. Evaluating')
    write_best_accuracy_to_file(accuracies, CONFIG.split('.json')[0] + '/validation_result.txt')
    obtain_predicts(mode='ru', test_mode='test', MAX_LEN=9999999, log_dict=True)


train()

