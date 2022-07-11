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
from deeppavlov import build_model
from deeppavlov.models.preprocessors.torch_transformers_preprocessor import TorchTransformersPreprocessor
from torch.distributions import Categorical
from scipy.stats import pearsonr
import torch

np.random.seed(1488666)

csv_file_dict = {'train': 'train.tsv', 'validation': 'val.tsv', 'test': 'test.tsv'}
try:
    SAMPLING = f'{sys.argv[1]}'
except:
    SAMPLING = 'plain'
try:
    MODE = f'{sys.argv[2]}'
except:
    MODE = 'englue'
print(f'Run with sampling {SAMPLING} mode {MODE}')

CONFIG = 'config_custom_glue.json'

model = build_model(CONFIG, download=False)

if not os.path.exists(SAMPLING):
    os.mkdir(SAMPLING)

if 'rusuperglue' in MODE:
    raise Exception('no pickled dataset - please obtain the one')
elif 'ensuperglue' in MODE:
    PICKLE_FILE = 'ensuperglue.pkl'
elif 'englue' in MODE:
    PICKLE_FILE = 'sampling/glue1.pkl'
else:
    raise Exception(f'Unsupported mode {MODE}')



BATCH_SIZE = 32
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



class Task:
    def __init__(self, name, filename, default_batch, classes, task_id):
        self.name = name
        self.filename = filename
        self.default_batch = default_batch
        self.classes = classes
        self.task_id = task_id

tasks = [Task('cola', 'CoLA.tsv', [0,1], 0),
	     Task('sst2', 'SST-2.tsv', [0, 1], 1),
	     Task('qqp', 'QQP.tsv', [0, 1], 2),
	     Task('mrpc', 'MRPC.tsv',[0, 1], 3),
	     Task('rte', 'RTE.tsv',  ['entailment', 'not_entailment'], 4),
	     Task('mnli-m', 'MNLI-m.tsv',  ['entailment', 'neutral', 'contradiction'], 5),
	     Task('mnli-mm', 'MNLI-mm.tsv',  ['entailment', 'neutral', 'contradiction'], 5),
	     Task('qnli', 'QNLI.tsv', ['entailment', 'not_entailment'], 6),
	     Task('stsb', 'STS-B.tsv', [1], 7),
	     Task('ax', 'AX.tsv', ['entailment', 'neutral','contradiction'], 5)]
tasks_to_train = [Task('cola', 'CoLA.tsv', [0,1], 0),
	              Task('sst2', 'SST-2.tsv',  [0, 1], 1),
	              Task('qqp', 'QQP.tsv',[0, 1], 2),
	              Task('mrpc', 'MRPC.tsv', [0, 1], 3),
	              Task('rte', 'RTE.tsv',  ['entailment', 'not_entailment'], 4),
	              Task('mnli', 'MNLI-m.tsv',  ['entailment', 'neutral', 'contradiction'], 5),
	              Task('qnli', 'QNLI.tsv', ['entailment', 'not_entailment'], 6),
	              Task('stsb', 'STS-B.tsv', [1], 7)]



def get_glue_metric(task,test_mode='test', mode=None, MAX_LEN=1000000, log_dict=True,submit_dir=''):
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
            

def obtain_glue_predicts(test_mode='test', mode=None, MAX_LEN=1000000, log_dict=True,submit_dir=''):
    if not os.path.exists(submit_dir):
        os.mkdir(submit_dir)
    accuracies = dict()
    for task in tasks[::-1]:  # dictionary:
        answer = get_glue_metric(task,test_mode=test_mode,MAX_LEN=MAX_LEN,log_dict=log_dict,submit_dir=submit_dir)
        if test_mode == 'validation':
            accuracies[task.name] = answer
    return accuracies

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


def sample_plain(epochs_done=0, return_probs=False,
                 task_ids_to_repeat = [i for i in data['train']]):
    # Равномерно сэмплируем
    global data
    probs = [len(data['train'][s]) for s in data['train']]

    for task_id in range(len(probs)):
        if task_id not in task_ids_to_repeat:
            probs[task_id] = 0

    tot = sum(probs)
    probs = [p / tot for p in probs]
    if return_probs:
        return probs
    return sample_with_probs(probs)


def sample_uniform(epochs_done=0, return_probs=False,
                   task_ids_to_repeat = [i for i in data['train']]):
    # Равномерно сэмплируем
    probs = [1.0 / NUM_TASKS for _ in range(NUM_TASKS)]
    for task_id in range(len(probs)):
        if task_id not in task_ids_to_repeat:
            probs[task_id] = 0

    tot = sum(probs)
    probs = [p / tot for p in probs]
  
    if return_probs:
        return probs
    return sample_with_probs(probs)


def sample_anneal(epochs_done=0, return_probs=False,
                  task_ids_to_repeat = [i for i in data['train']]):
    # annealed sampling
    global data
    sizes = [len(data['train'][s]) for s in data['train']]
    alpha = 1.0 - 0.8 * (epochs_done / NUM_TRAIN_EPOCHS)
    # print('Sizes ')
    # print(self._get_data_size(data))
    probs = [p ** alpha for p in sizes]
    for task_id in range(len(probs)):
        if task_id not in task_ids_to_repeat:
            probs[task_id] = 0
    assert sum(probs) > 0, breakpoint()
    tot = sum(probs)
    probs = [p / tot for p in probs]
    if return_probs:
        return probs
    return sample_with_probs(probs)


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
preprocessor_index = None
for i in range(len(model.pipe)):
    if MODEL_NAME in str(model.pipe[i][-1]):
        model_index = i
for i in range(len(model.pipe[model_index])):
    if 'MultiTaskPipelinePreprocessor' in str(model.pipe[model_index][i]):
        preprocessor_index = i
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


sizes = [len(data['train'][s]) for s in data['train']]
accuracies = dict()
durations = dict()
curr_durations = dict()
for task in tasks_to_train:
    accuracies[task.name] = []
    durations[task.name] = []
    curr_durations[task.name] = 0



ALPHA = 0.9
NUM_TASK_EPOCHS = 5
EVERY_TASK_EARLY_STOP_PATIENCE=2
if 'plain' in SAMPLING.lower():
    SAMPLING_FUNCTION=sample_plain
elif 'uniform' in SAMPLING.lower():
    SAMPLING_FUNCTION=sample_uniform
elif 'anneal' in SAMPLING.lower():
    SAMPLING_FUNCTION=sample_anneal

def write_accuracies_durations(task_ids = [i for i in range(len(tasks_to_train))],
                               ACCURACY_DURATION_FILE = f'{ALPHA}_accuracies_durations.txt'):
    for task_id in task_ids:
        task = tasks_to_train[task_id]
        if task.name != 'mnli':
            accuracies[task.name].append(get_glue_metric(task,test_mode='validation',MAX_LEN=MAX_LEN,log_dict=log_dict))
        else:
            curr_tasks = [Task('mnli-m', 'MNLI-m.tsv',  ['entailment', 'neutral', 'contradiction'], 0),
		     Task('mnli-mm', 'MNLI-mm.tsv',  ['entailment', 'neutral', 'contradiction'], 0)]
            curr_accuracies = [get_glue_metric(curr_task,test_mode='validation',
                                               MAX_LEN=MAX_LEN,log_dict=log_dict) for curr_task in curr_tasks]
            accuracies[task.name].append(sum(accuracies)/len(accuracies))
        durations[task.name].append(curr_durations[task.name])
    durations['ids_to_repeat'] = task_ids_to_repeat
    log_file = open(ACCURACY_DURATION_FILE,'w')
    log_file.write(str((accuracies,durations)))
    log_file.close()


def not_early_stop(list_of_accuracies, patience=EVERY_TASK_EARLY_STOP_PATIENCE):
    if len(list_of_accuracies)>patience:
        if all([list_of_accuracies[-k]<max(list_of_accuracies) for k in range(1,patience+1)]):
            return False
    return True

def make_name():
    FUNCTION_NAME = str(SAMPLING_FUNCTION).split('function ')[1].split(' at')[0]
    CONFIG_NAME = CONFIG.split('.json')[0]
    return 'predicts/{ALPHA}{CONFIG_NAME}{FUNCTION_NAME}'



def train(log=True, sampling_function =SAMPLING_FUNCTION,log_dict=False,debug=False):
    task_ids_to_repeat = []
    batches_per_epoch = [1+(size //BATCH_SIZE) for size in sizes]
    task_ids_to_iterate = np.argsort(sizes)[::-1]
    write_accuracies_durations()
    for task_id in task_ids_to_iterate:
        task = tasks_to_train[task_id]
        write_accuracies_durations()
        for epoch in range(NUM_TASK_EPOCHS):
            model.pipe[model_index][-1].load()
            task_accuracies = accuracies[task.name]
            if not_early_stop(task_accuracies):
                num_batches_required = batches_per_epoch[task.task_id]
                num_batches_passed = 0
                while num_batches_passed < num_batches_required:
                    p = random.random()
                    if p < ALPHA or len(task_ids_to_repeat) == 0:
                        if log_dict:
                            print(f'Sampling from {task.name}')
                        _args = sampling_function(task_ids_to_repeat=task)
                        num_batches_passed +=1
                    else:
                        _args = sampling_function(task_ids_to_repeat=task_ids_to_repeat)
                        # we dont increment num_batches_passed
                    _args_to_train = model.pipe[model_index][preprocessor_index](_args)
                    breakpoint()
                    if not debug or p <0.0005:  # on debug, we almost skip training
                        losses = model.pipe[model_index][-1].train_on_batch(*(_args_to_train))
                    for task_index in range(NUM_TASKS):
                        curr_durations[task_index] += len(_args[task_index])
                write_accuracies_durations()
                if task_accuracies[-1]==max(task_accuracies):
                    model.pipe[model_index][-1].save()
            else:
                print(f'Early stopping. Epoch {epoch} skipped')
        task_ids_to_repeat.append(task_id)
    print('Finishing!')
    model.pipe[model_index][-1].load()
    obtain_predicts(mode='ru', test_mode='test', MAX_LEN=9999999, log_dict=True,submit_dir=make_name())

train()
