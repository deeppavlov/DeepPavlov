#Using script: python deeppavlov/utils/glue_eval/evaluate_glue_superglue.py PATH_TO_CONFIG DATASET_TYPE DIRECTORY_TO_WRITE_SUBMIT TASK_ID
#If the model is multilabel don't provide TASK_ID. Otherwise, TASK_ID must be exactly as the field task_id that corresponding object of type Task has
#Outputs of model ( first N_TASK outputs ) should be probabilities, except for sts-b. Order of tasks must be exactly the same as in the original configs

import json
import logging
import os
import sys
from collections import defaultdict, namedtuple
from copy import deepcopy
from typing import Iterable
import argparse

import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from deeppavlov import build_model
from deeppavlov.dataset_readers.huggingface_dataset_reader import add_num_examples, preprocess_multirc, \
    preprocess_record, preprocess_wsc

np.random.seed(282)

parser = argparse.ArgumentParser()
parser.add_argument('config_name',type=str, required=True,
                    help='Name of config to evaluate')
parser.add_argument('dataset_type', type=str,required=True,
                    choices=['glue','rusuperglue','ensuperglue','superglue','all'],
                    help='Type of the dataset to evaluate')
parser.add_argument('submit_dir', type=str,required=True,
                    help='Directory to submit predicts')
parser.add_argument('task_id',nargs='?', const=1,type=int, default=None)
                    help='Id of task to evaluate. None if multitask model used')
parser.add_argument('max_valid_samples_by_task',nargs='?', const=1,type=int, default=100000,
                    help='Max valid samples by task to perform the evaluation on')
parser.add_argument('model_name',nargs='?', const=1,type=str, default='MultiTaskBert',
                    help='Max valid samples by task to perform the evaluation on')
args = parser.parse_args()

print('Building model')
logging.warning(f'Before evaluating, make sure that a) model returns probabilities first b) the order of task-specific arguments, if model is multitask, is exactly as in this script')
model = build_model(args.config_name, download=True)
model_index = None
for i in range(len(model.pipe)):
    if args.model_name in str(model.pipe[i][-1]):
        model_index = i
if model_index == None:
    raise Exception(f'{args.model_name} not found in {model.pipe}')
model.pipe[model_index][-1].load()

Task = namedtuple('Task', 'name filename classes task_id')

glue_tasks = [Task('cola', 'CoLA.tsv', [0,1], 0),
            Task('sst2', 'SST-2.tsv', [0, 1], 1),
            Task('qqp', 'QQP.tsv', [0, 1], 2),
            Task('mrpc', 'MRPC.tsv',[0, 1], 3),
            Task('rte', 'RTE.tsv',  ['entailment', 'not_entailment'], 4),
            Task('mnli-m', 'MNLI-m.tsv',  ['entailment', 'neutral', 'contradiction'], 5),
            Task('mnli-mm', 'MNLI-mm.tsv',  ['entailment', 'neutral', 'contradiction'], 5),
            Task('qnli', 'QNLI.tsv', ['entailment', 'not_entailment'], 6),
            Task('stsb', 'STS-B.tsv', [1], 7),
            Task('ax', 'AX.tsv', ['entailment', 'neutral','contradiction'], 5)]
glue_tasks_to_train = [Task('cola', 'CoLA.tsv', [0,1], 0),
                    Task('sst2', 'SST-2.tsv',  [0, 1], 1),
                    Task('qqp', 'QQP.tsv',[0, 1], 2),
                    Task('mrpc', 'MRPC.tsv', [0, 1], 3),
                    Task('rte', 'RTE.tsv',  ['entailment', 'not_entailment'], 4),
                    Task('mnli', 'MNLI-m.tsv',  ['entailment', 'neutral', 'contradiction'], 5),
                    Task('qnli', 'QNLI.tsv', ['entailment', 'not_entailment'], 6),
                    Task('stsb', 'STS-B.tsv', [1], 7)]
rusuperglue_tasks = [Task('rwsd', 'RWSD.jsonl', ["False", "True"], 0),
             Task('muserc', 'MuSeRC.jsonl', [0, 1], 1),
             Task('rcb', 'RCB.jsonl', ['contradiction', 'neutral', 'entailment'], 2),
             Task('rucos', 'RuCoS.jsonl', [1, 0], 3),
             Task('danetqa', 'DaNetQA.jsonl', ['false', 'true'], 4),
             Task('parus', 'PARus.jsonl', [0, 1], 5),
             Task('terra', 'TERRa.jsonl', ['entailment', 'not_entailment'], 6),
             Task('russe', 'RUSSE.jsonl', ['false', 'true'], 7),
             Task('lidirus', 'LiDiRus.jsonl', ['not_entailment', 'entailment'], 6)]
ensuperglue_tasks = [Task('wsc', 'WSC.jsonl', ["False", "True"], 0),
             Task('multirc', 'MultiRC.jsonl', [0, 1], 1),
             Task('cb', 'CB.jsonl', ['entailment','contradiction','neutral'], 2),
             Task('record', 'ReCoRD.jsonl', [0, 1], 3),
             Task('boolq', 'BoolQ.jsonl', ['false', 'true'], 4),
             Task('copa', 'COPA.jsonl', [0, 1], 5),
             Task('rte', 'RTE.jsonl', ['entailment', 'not_entailment'], 6),
             Task('wic', 'WiC.jsonl', ['false', 'true'], 7),
             Task('axb', 'AX-b.jsonl', ['entailment', 'not_entailment'], 6)]

eval_tasks = ['axb','lidirus','ax']  # Tasks only for evaluation

if args.dataset_type == 'glue':
    logging.info('GLUE task set')
    tasks = deepcopy(glue_tasks)
    tasks_to_train = deepcopy(glue_tasks_to_train)
elif args.dataset_type=='rusuperglue':
    logging.info('RUSUPERGLUE task set')
    tasks = rusuperglue_tasks
    tasks_to_train = deepcopy(tasks[:-1])
elif args.dataset_type=='ensuperglue' or dataset_type=='superglue':
    logging.info('ENSUPERGLUE task set')
    dataset_type='super_glue' # backward compatibility
    tasks = ensuperglue_tasks
    tasks_to_train = deepcopy(tasks[:-1])
elif args.dataset_type=='all':
    tasks = deepcopy(ensuperglue_tasks + rusuperglue_tasks)
    num_superglue_tasks = len(ensuperglue_tasks) - 1  # 8
    for i in range(len(ensuperglue_tasks), len(tasks)):
        tasks[i].task_id = tasks[i].task_id + num_superglue_tasks
    tasks_to_train = tasks[:len(ensuperglue_tasks)-1] + tasks[len(ensuperglue_tasks):-1]

args.SUBMIT_DIR = sys.argv[3]
if args.task_id is not None:
    tasks = [task for task in tasks if task.task_id == args.task_id]
    for i in range(len(tasks)):
        tasks[i].task_id=0
    tasks_to_train = [task for task in tasks if task.name not in eval_tasks]
    print(f'Assuming singletask setting')
else:
    print('Assuming multitask setting')
    task_id = None
NUM_TASKS = len(tasks_to_train)

print(f'Assuming that tasks to evaluate are {[k.name for k in tasks]}')
print(f'Assuming that tasks to train are {[k.name for k in tasks_to_train]}')
print('Check the order of tasks carefully')


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

def get_superglue_metric(task, log_dict=False,submit_dir='',split='test',
                              default_batch_size=1):
    global model, dataset_type
    accuracies = {}
    def get_prediction(classes, i, batch, return_probas=False):
        x=[[] for _ in range(NUM_TASKS)]
        y=[[] for _ in range(NUM_TASKS)]
        x[i]= batch
        predictions = model(*(x+y))
        predicted_probas = predictions[i][0]
        if return_probas:
            return predicted_probas
        else:
            predicted_class = np.argmax(predicted_probas)
            prediction = classes[predicted_class]
            return prediction
      
    counter = defaultdict(int)
    true_count = 0
    total_count = 0
    true_to_total = defaultdict(lambda: defaultdict(int))
    label_counter = defaultdict(int)
    filename = f'{submit_dir}/{task.filename}'
    output = []
    dataset = load_dataset(dataset_type, task.name, split=split)
    if task.name in ['rwsd','wsc']:
        dataset = dataset.map(preprocess_wsc,batched=True,remove_columns=["span1_index", "span2_index", "span1_text", "span2_text"])
    elif task.name in ['muserc','multirc']:
        dataset = dataset.map(preprocess_multirc, batched=True, remove_columns=["paragraph", "question"])
        muserc_output = {}
    elif task.name in ['rucos','record']:
        dataset = dataset.map(preprocess_record, batched=True,
                              remove_columns=["answers"]).map(add_num_examples, batched=True, batch_size=None)
        record_predictions = []
        record_indexes = []
        record_entities = []
        record_output = dict()

    loader = DataLoader(dataset, batch_size=1)
    index_list,entities_list,predictions=[],[],[] # for rucos
    printed_first_batch=False
    k=0
    for batch in tqdm(loader):
        k+=1
        if not printed_first_batch:
            print(batch)
            printed_first_batch=True
        index = batch['idx']
        if task.name in ['terra', 'rte', 'rcb', 'cb']:
            TASK_BATCH = [(batch['premise'][0],batch['hypothesis'][0])]
        elif task.name in ['lidirus','axb', 'wic','russe']:
            TASK_BATCH=[(batch['sentence1'][0],batch['sentence2'][0])]
        elif task.name in ['danetqa','boolq']:
            TASK_BATCH = [(batch['question'][0],batch['passage'][0])]            
        elif task.name in ['parus','copa']:
            if task == 'parus':
                question_dict = {
                    "cause": "Что было причиной этого?",
                    "effect": "Что случилось в результате?"
                }
            else:
                question_dict = {
                    "cause": "What was the cause of this?",
                    "effect": "What happened as a result?"}
            question = question_dict[batch["question"][0]]
            context = f"{batch['premise'][0]} {question}"
            choices = [batch["choice1"][0], batch["choice2"][0]]
            TASK_BATCH = [(context, choices)]
        elif task.name in ['rwsd', 'wsc']:
            index = batch["idx"][0]
            context = batch["text"][0]
            answer = batch["answer"][0]
            TASK_BATCH = [(context, answer)]               
        elif task.name in ['muserc','multirc']:
            indices = batch["idx"]
            paragraph_idx = int(indices["paragraph"][0])
            question_idx = int(indices["question"][0])
            answer_idx = int(indices["answer"][0])
            context = batch["context"][0]
            answer = batch["answer"][0]
            TASK_BATCH = [(context, answer)]
        elif task.name in ['rucos','record']:
            indices = batch["idx"]
            queries = batch["query"]
            passages = batch["passage"]
            entities = batch["entities"]
            num_examples = batch["num_examples"]
            TASK_BATCH = [(id, query, passage, entity, n)
                          for id, query, passage, entity, n in
                          zip(indices, queries, passages, entities, num_examples)]
        prediction = get_prediction(task.classes, task.task_id, 
                                    TASK_BATCH, 
                                    return_probas = task.name in ['rucos','record'])
        true_class = task.classes[int(batch['label'][0].int())]
        if task.name in ['muserc', 'multirc']: # postprocess
            if paragraph_idx not in muserc_output:
                muserc_output[paragraph_idx] = dict(
                    idx=paragraph_idx, passage=dict(
                        questions=[
                            dict(
                                idx=question_idx,
                                answers=[
                                    dict(
                                        idx=answer_idx,
                                        label=prediction
                                    )
                                ]
                            )
                        ]
                    )
                )
            questions = muserc_output[paragraph_idx]["passage"]["questions"]
            question_indices = set(el["idx"] for el in questions)
            if question_idx not in question_indices:
                muserc_output[paragraph_idx]["passage"]["questions"].append(dict(
                    idx=question_idx, answers=[dict(idx=answer_idx, label=prediction)]
                ))
            else:
                for question in muserc_output[paragraph_idx]["passage"]["questions"]:
                    if question["idx"] == question_idx:
                        question["answers"].append(dict(idx=answer_idx, label=prediction))
        else:
            if task.name in ['rucos', 'record']:
                prediction = float(prediction[0])
                record_predictions += [prediction]
                record_indexes += indices
                record_entities += entities
            else:
                output.append(dict(idx=int(index), label=prediction)) # universal?
        if split == 'validation':
            if isinstance(prediction, float) and task.name not in ['rucos', 'record']:
                prediction = int(round(prediction, 0)) # for float-like preds for RECORD
            true_to_total[prediction][true_class] += 1
            if prediction == true_class:
                true_count += 1
            total_count += 1
            accuracies[task] = true_count / total_count
            if total_count >= MAX_VALID_SAMPLES_BY_TASK:
                break
    # postprocess for record
    
    if split == 'validation':
        print('Accuracy')
        print(true_count/total_count)
    else:
        if task.name in ['rucos', 'record']:
            for index, entity, prediction in zip(record_indexes, record_entities, record_predictions):
                if index not in record_output:
                    record_output[index] = {'predicted': [], 'probability': []}
                record_output[index]["predicted"].append(entity)
                record_output[index]["probability"].append(float(prediction))
            output = []
            for key, value in record_output.items():
                answer_index = np.argmax(value["probability"])
                answer = value["predicted"][answer_index]
                output.append(
                    {
                        "idx": int(key.split("-")[-2]),
                        "label": answer
                    })
        elif task.name in ['muserc', 'multirc']:
            output = list(muserc_output.values())
        with open(filename, "w") as file:
            for element in sorted(output, key=lambda d: d["idx"]):
                file.write(json.dumps(element, ensure_ascii=False, cls=NpEncoder) + "\n")  
    if log_dict:
        print(counter)
        print(true_to_total)

def get_glue_metric(task,split='test',log_dict=True,submit_dir=''):
    look_name = task.name.split('-m')[0]
    if 'mnli' not in task.name:
        name = task.name
    elif task.name == 'mnli-m':
        name = 'mnli_matched'
        split = split + '_matched'
    elif task.name == 'mnli-mm':
        name = 'mnli_mismatched'
        split = split + '_mismatched'
    dataset = load_dataset("glue", look_name, split=split)

    loader = DataLoader(dataset, batch_size=1)
    predictions = []
    labels = []
    k=0
    pred_to_true = defaultdict(lambda: defaultdict(int))
    for batch in tqdm(loader):
        if k==0:
            print(batch)
            k+=1
        if k==MAX_VALID_SAMPLES_BY_TASK and 'test' not in split:
            break
        if task.name in ['cola','sst2']:
            examples = [j for j in batch['sentence']]
        elif task.name in ['rte', 'axb', 'stsb', 'mrpc']:
            examples = [(sentence1, sentence2)
                        for sentence1, sentence2 in zip(batch['sentence1'], batch['sentence2'])]
        elif task.name == 'qqp':
            examples = [(sentence1, sentence2)
                        for sentence1, sentence2 in zip(batch['question1'], batch['question2'])] 
        elif task.name == 'qnli':
            examples = [(sentence1, sentence2)
                        for sentence1, sentence2 in zip(batch['question'], batch['sentence'])] 
        elif task.name in ['mnli-m', 'mnli-mm','mnli', 'ax']:
            examples = [(sentence1, sentence2)
                        for sentence1, sentence2 in zip(batch['premise'], batch['hypothesis'])]
        else:
            raise Exception(F'Unsupported taskname {task.name}')

        batch_to_use = [[] for _ in tasks_to_train] + [[] for _ in tasks_to_train]
        batch_to_use[task.task_id] = examples
        pred = model(*batch_to_use)[task.task_id]
        k+=1
        if task.name == 'stsb':
            if isinstance(pred, Iterable):
                pred = float(pred[0])
            new_pred = min(5, max(0,float(pred)))
            predictions = predictions + [float(new_pred)]
        else:
            new_pred = [np.argmax(pred)]
            predictions = predictions + [task.classes[int(k)] for k in new_pred]
        if split != 'test':
            if task.name=='stsb':
                labels = labels + list([float(k) for k in batch['label']])
            else:
                labels = labels + list([task.classes[int(s)] for s in batch['label']])
    print(predictions[:10])
    print(labels[:10])
    if 'test' not in split:
        for prediction, label in zip(predictions, labels):
            pred_to_true[prediction][label]+=1
        if task.name == 'stsb':
            metric = pearsonr(predictions, labels)[0]
        else:
            metric = accuracy_score(predictions, labels)
        print(f'Metric is {metric}')
        return metric
    if log_dict and task.name != 'stsb':
        from collections import Counter
        print(Counter(predictions))
    default_pred = pd.DataFrame({'predictions': predictions})
    default_pred.to_csv(f'{submit_dir}/{task.filename}', sep='\t')


def obtain_predicts(task,dataset_type,log_dict=True,submit_dir='',split='test'):
    if not os.path.exists(submit_dir):
        os.mkdir(submit_dir)
    if dataset_type=='glue':
        answer = get_glue_metric(task, log_dict=log_dict,submit_dir=submit_dir,split=split)
    else:
        answer = get_superglue_metric(task, log_dict=log_dict,submit_dir=submit_dir,split=split)

for task in tasks:
    if task.name not in ['ax','axb']:
        splits = ['validation', 'test']
    else:
        splits = ['test']
    for split in splits:
        print(f'Evaluating {task.name} on the {split} set')
        obtain_predicts(task, dataset_type,log_dict=True,submit_dir=args.submit_dir, split=split)
