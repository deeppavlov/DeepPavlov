# logits output shape [batch_size,max_seq_len,N_CLASSES]

#input ids token type ids attention mask [batch_size,max_seq_len]
# labels - INITIAL batch_size [batch_size,max_seq_len1] where max_seq_len1 < max_seq_len because max_seq_len1 is about words, while all else about subwords

#input for NER preprocessor - tuple(first list of tokens,second list of tokens)...
#list of tokens is sentence


from logging import getLogger
from typing import List, Dict, Union, Optional
from pathlib import Path
import math 
import copy
import json
import six
import numpy as np
from overrides import overrides
import os
import _pickle as cPickle

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.autograd import Variable
from collections.abc import Iterable
from torch.nn.parameter import Parameter
from transformers import AutoConfig, AutoModelForSequenceClassification

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.models.torch_bert.torch_transformers_sequence_tagger import token_from_subtoken, token_labels_to_subtoken_labels

log = getLogger(__name__)


prev_input = None 
class BertForMultiTask(nn.Module):
    """BERT model for classification or regression on GLUE tasks (STS-B is treated as a regression task).
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    ```
    """

    def __init__(self, tasks_num_classes,
                 backbone_model='bert_base_uncased', config_file=None,
                 max_seq_len=320):

        super(BertForMultiTask, self).__init__()
        config = AutoConfig.from_pretrained(backbone_model,output_hidden_states=True,output_attentions=True)
        self.bert = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=backbone_model,
                                                                           config=config )
        self.classes = tasks_num_classes  # classes for every task
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.max_seq_len =max_seq_len
        self.activation = nn.Tanh()
        OUT_DIM = config.hidden_size
        self.bert.final_classifier = nn.ModuleList(
            [
                nn.Linear(OUT_DIM, num_labels) for num_labels in self.classes
            ]
        )
        self.bert.pooler = nn.Linear(OUT_DIM, OUT_DIM)
        

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        task_id,
        name="classification",
        labels=None,
        span1=None,
        span2=None
    ):
        outputs=None
        if name in ['sequence_labeling', 'multiple_choice']:
            # delete after checking label format
            input_ids = input_ids.view(-1, input_ids.size(-1))
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        try:
            outputs = self.bert(input_ids=input_ids.int(),
                                token_type_ids=token_type_ids.int(),
                                 attention_mask=attention_mask.int())
        except Exception as e:
            breakpoint()
            raise e
        last_hidden_state = outputs[1][-1]
        first_token_tensor = last_hidden_state[:, 0]        
        pooled_output = self.bert.pooler(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        if name == 'sequence_labeling':
            final_output = self.dropout(last_hidden_state)
            logits = self.bert.final_classifier[task_id](final_output)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                active_logits = logits.view(-1, self.classes[task_id])
                loss=loss_fct(active_logits, labels.view(-1))
                return loss, logits     
            else:
                return logits
        elif name in ['classification','regression', 'multiple_choice']:
            pooled_output = self.dropout(pooled_output)
            logits = self.bert.final_classifier[task_id](pooled_output)
            if name=='multiple_choice':
                logits = logits.view((-1,self.classes[task_id]))
                if labels is not None:
                    assert len(logits)==len(labels),breakpoint()
            if labels is not None:
                if name != "regression":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits, labels.view(-1))
                    return loss, logits
                elif  name == "regression":
                    loss_fct = MSELoss()
                    logits=logits.cpu()
                    labels=labels.cpu()
                    try:
                        loss = loss_fct(logits, labels.unsqueeze(1))
                    except:
                        breakpoint()
                    return loss, logits
            else:
                return logits
        elif name == 'span_classification':
            raise Exception('Span classification not yet supported')
            assert span1 is not None and span2 is not None
            log.warning('SPAN CLASSIFICATION IS SUPPORTED ONLY EXPERIMENTALLY')
            input_dict = {"input":last_hidden_state, "attention_mask": attention_mask, 
                          "span1s":span1, "span2s":span2}
            if labels is not None:
                input_dict["labels"] = labels
            output_dict = self.bert.final_classifier[task_id].forward(input_dict)
            if 'loss' in output_dict:
                return output_dict['loss'], output_dict['logits']
            else:
                return output_dict['logits']
        elif name == 'question_answering':
            raise Exception('Question answering not yet supported')      
            sequence_output = self.dropout(last_hidden_state)
            # or logits?
            start_logits, end_logits = last_hidden_state.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            if labels is not None:
            # If we are on multi-GPU, split add a dimension - if not this is a no-op
                start_positions, end_positions = labels
                start_positions = start_positions.squeeze(-1)
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)
                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)
                total_loss = (start_loss + end_loss) / 2
                return total_loss
            else:
                return start_logits, end_logits
            

@register('multitask_bert')
class TorchMultiTaskBert(TorchModel):
    """Multi-Task Bert Based Model
    Args:
        tasks: Dict of task names along with the labels for each task,
        pretrained_bert: path of the pretrained bert embeddings
        freeze_embeddings: set True if bert embeddings are to be freezed,
        optimizer: optimizer name defaults to AdamW,
        optimizer_parameters: optimizer parameters,
        lr_scheduler: name of the lr scheduler,
        lr_scheduler_paramters: lr scheduler parameters for the scheduler,
        gradient_accumulation_steps: number of gradient accumulation steps,
        steps_per_epoch: number of steps taken per epoch
        clip_norm: normalization: value for gradient clipping,
        one_hot_labels: set to true if using one hot labels,
        multilabel: set true for multilabel class,
        return_probas: set true to return prediction probas,
        in_distribution: in_distribution: The distribution of variables listed
        in the ``"in"`` config parameter between tasks.
        ``in_distribution`` can be ``None`` if only 1 task is called.
        In that case all variableslisted in ``"in"`` are arguments of 1 task.
        ``in_distribution`` can be a dictionary of ``int``. If that is the
        case, then keys of ``in_distribution`` are task names and values are
        numbers of variables from ``"in"`` parameter of config which are inputs
        of corresponding task. The variables in ``"in"`` parameter have to be
        in the same order the tasks are listed in ``in_distribution``.
        in_y_distribution: Same as ``in_distribution`` for ``"in_y"`` config parameter.,
    """

    def __init__(
        self,
        tasks: Dict[str, Dict],
        pretrained_bert: str = None,
        freeze_embeddings: bool = False,
        max_seq_len: str = 320,
        optimizer: str = "AdamW",
        optimizer_parameters: dict = {"lr": 2e-5},
        config: str = "configs/top_config.json",
        lr_scheduler: Optional[str] = None,
        lr_scheduler_parameters: dict = {},
        gradient_accumulation_steps: Optional[int] = 1,
        steps_per_epoch: Optional[int] = None,
        backbone_model:str="bert-base-cased",
        clip_norm: Optional[float] = None,
        one_hot_labels: bool = False,
        multilabel: bool = False,
        return_probas: bool = False,
        flatten_multiplechoice_labels: bool=True,
        in_distribution: Optional[Union[Dict[str, int], Dict[str, List[str]]]] = None,
        in_y_distribution: Optional[Union[Dict[str, int], Dict[str, List[str]]]] = None,
        *args,
        **kwargs,
    ) -> None:
        path_to_current_file = os.path.realpath(__file__)
        current_directory = os.path.split(path_to_current_file)[0]
        self.config = os.path.join(
            current_directory, config)
        self.return_probas = return_probas
        self.one_hot_labels = one_hot_labels
        self.multilabel = multilabel
        self.clip_norm = clip_norm
        self.task_names = list(tasks.keys())
        self.tasks_type = []
        self.backbone_model = backbone_model
        self.max_seq_len=max_seq_len
        self.tasks_num_classes = []
        self.task_names = []
        for task in tasks:
            self.task_names.append(task)
            self.tasks_num_classes.append(tasks[task]['options'])
            self.tasks_type.append(tasks[task]['type'])
        self.n_tasks = len(tasks)
        self.train_losses = [[] for task in self.task_names]
        self.pretrained_bert = pretrained_bert
        self.freeze_embeddings = freeze_embeddings
        self.optimizer_name = optimizer
        self.optimizer_parameters = optimizer_parameters
        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_parameters = lr_scheduler_parameters
        self.gradient_accumulation_steps = [gradient_accumulation_steps for _ in self.task_names]
        self.steps_per_epoch = steps_per_epoch
        self.in_distribution = in_distribution
        self.in_y_distribution = in_y_distribution        
        if not self.in_distribution:
            self.in_distribution = {task: 1 for task in tasks}
        if not self.in_y_distribution:
            self.in_y_distribution = {task: 1 for task in tasks}
        self.steps_taken = 0
        self.prev_id = None
        self.printed = False
        if self.multilabel and not self.one_hot_labels:
            raise RuntimeError(
                "Use one-hot encoded labels for multilabel classification!"
            )

        if self.multilabel and not self.return_probas:
            raise RuntimeError(
                "Set return_probas to True for multilabel classification!"
            )

        assert not self.multilabel, 'Multilabel not supported yet'
        super().__init__(
            optimizer_parameters=self.optimizer_parameters,
            lr_scheduler=self.lr_scheduler_name,
            lr_scheduler_parameters=self.lr_scheduler_parameters,
            **kwargs,
        )

    @overrides
    def init_from_opt(self) -> None:
        """Initialize from scratch `self.model` with the architecture built
        in `model_func (MultitaskBert)` method of this class along with
        `self.optimizer` as `self.optimizer_name` from `torch.optim` and
        parameters `self.optimizer_parameters`, optionally initialize
        `self.lr_scheduler` as `self.lr_scheduler_name` from
        `torch.optim.lr_scheduler` and parameters `self.lr_scheduler_parameters`
        """

    
        self.model = BertForMultiTask(
                backbone_model=self.backbone_model,
                tasks_num_classes = self.tasks_num_classes)
        self.model = self.model.to(self.device)
        no_decay = ["bias", "gamma", "beta"]
        base = ["attn"]
        get_non_decay_params = lambda model: [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in base)
                ]
        get_decay_params = lambda model: [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in base)
                ]
        model_parameters = [
            {
                "params": get_non_decay_params(self.model),
                "weight_decay": 0.01,
            },
            {
                "params": get_decay_params(self.model),
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = getattr(torch.optim, self.optimizer_name)(
            model_parameters, **self.optimizer_parameters
        )

        if self.lr_scheduler_name:
            self.lr_scheduler = getattr(
                torch.optim.lr_scheduler, self.lr_scheduler_name
            )(self.optimizer, **self.lr_scheduler_parameters)
            breakpoint()

    @overrides 
    def load(self, fname: Optional[str] = None) -> None:
        if fname is not None:
            self.load_path = fname

        if self.load_path:
            log.info(f"Load path {self.load_path} is given.")
            if isinstance(
                    self.load_path,
                    Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            weights_path = Path(self.load_path.resolve())
            weights_path = weights_path.with_suffix(f".pth.tar")
            if weights_path.exists():
                log.info(f"Load path {weights_path} exists.")
                log.info(
                    f"Initializing `{self.__class__.__name__}` from saved.")

                # firstly, initialize with random weights and previously saved
                # parameters
                self.init_from_opt()

                # now load the weights, optimizer from saved
                log.info(f"Loading weights from {weights_path}.")
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(
                    checkpoint["optimizer_state_dict"])
                self.epochs_done = checkpoint.get("epochs_done", 0)
            else:
                log.info(
                    f"Init from scratch. Load path {weights_path} does not exist.")
                self.init_from_opt()
        else:
            log.info(
                f"Init from scratch. Load path {self.load_path} is not provided.")
            self.init_from_opt()

        if self.freeze_embeddings:
            for n, p in self.model.bert.named_parameters():
                if (
                    "aug" in n
                    or "classifier" in n
                    or "mult" in n
                    or "gamma" in n
                    or "beta" in n
                ):
                    continue
                p.requires_grad = False
                log.info("Bert Embeddings Freezed")
    def _make_input(self,task_features,task_id,labels=None):
        batch_input_size = None
        print(f'Making input from {task_features}')
        if len(task_features) == 1 and isinstance(task_features,list):
            task_features = task_features[0]

        if isinstance(labels,Iterable) and all([k is None for k in labels]):
            labels=None
        _input = {}
        element_list = ["input_ids", "attention_mask", "token_type_ids"]

        if self.tasks_type[task_id] == 'span_classification':
            element_list = element_list + ['span1', 'span2']
        for elem in element_list:
            if elem in task_features:
                _input[elem] = task_features[elem]
                batch_input_size = _input[elem].shape[0]
            elif hasattr(task_features,elem):
                _input[elem] = getattr(task_features,elem)
                batch_input_size = _input[elem].shape[0]
            if elem in _input:           
                if self.tasks_type[task_id] in ['sequence_labeling','multiple_choice']:
                    _input[elem] = _input[elem].view((-1, _input[elem].size(-1)))

        if labels is not None:
            print(f'Using {labels}')
            if self.tasks_type[task_id] == "regression":
                _input["labels"] = torch.tensor(
                    np.array(labels, dtype=float), dtype=torch.float32
                )
            elif self.tasks_type[task_id] == 'multiple_choice':
                labels = torch.Tensor(labels).long()
                onehot_labels = torch.nn.functional.one_hot(labels,num_classes=self.tasks_num_classes[task_id]) 
                _input['labels'] = torch.flatten(onehot_labels, start_dim=0)
            elif self.tasks_type[task_id] == 'sequence_labeling':
                subtoken_labels = [token_labels_to_subtoken_labels(y_el, y_mask, input_mask)
                           for y_el, y_mask, input_mask in zip(labels,_input['token_type_ids'].numpy(),
                                                               _input['attention_mask'].numpy())]
                _input['labels'] = torch.from_numpy(np.array(subtoken_labels)).to(torch.int64)
            else:
                _input["labels"] = torch.from_numpy(np.array(labels))
        for elem in _input:
            _input[elem] = _input[elem].to(self.device)            
        print('Made input on device')
        print(_input)
        if _input == {}:
            breakpoint()
            raise Exception('EMPTY INPUT!!!!')
        if 'labels' in _input:
            assert len(_input['labels'])==len(_input['input_ids']),breakpoint()

        return _input,batch_input_size


    def __call__(self, *args):
        """Make prediction for given features (texts).
        Args:
            features: batch of InputFeatures for all tasks
        Returns:
            predicted classes or probabilities of each class
        """
        ### IMPROVE ARGS CHECKING AFTER DEBUG
        self.validation_predictions = [None for _ in range(len(args))]
        for task_id in range(len(self.task_names)):
            if len(args[task_id]):
                _input, batch_input_size = self._make_input(task_features=args[task_id],task_id=task_id)
                assert 'input_ids' in _input, breakpoint()
                with torch.no_grad():
                    logits = self.model(
                        task_id=task_id, name=self.tasks_type[task_id], **_input
                    )
                #if task_id == 4:
                #    print(f'{_input["input_ids"].shape} {_input["token_type_ids"].shape} {logits.shape}')
                #    breakpoint()
                if self.tasks_type[task_id] == 'sequence_labeling':
                    if self.return_probas:
                        log.warning(f'Return_probas for sequence_labeling not supported yet. Returning ids for this task')
                    y_mask = _input['token_type_ids'].cpu()
                    logits = token_from_subtoken(logits.cpu(),y_mask)
                    predicted_ids = torch.argmax(logits, dim=-1).int().tolist()
                    seq_lengths = torch.sum(y_mask, dim=1).int().tolist()
                    pred=[prediction[:max_seq_len] for max_seq_len,prediction in zip(seq_lengths, predicted_ids)]
                    print(pred[0][0])
                elif self.tasks_type[task_id] == 'regression':
                    pred = logits[:,0]
                    assert len(pred) == len(_input['input_ids']), breakpoint()
                elif self.return_probas:
                    pred = torch.nn.functional.softmax(logits, dim=-1)
                else:
                    pred = torch.nn.functional.argmax(logits, dim=1)

                if not isinstance(pred, list):
                    print(f"{pred.shape}")  # delete this string later
                    pred = pred.tolist()
                self.validation_predictions[task_id] = pred
        if len(args) ==1:
            print(f'Predictions {self.validation_predictions[0]}')
            return self.validation_predictions[0]
        for i in range(len(self.validation_predictions)):
            if self.validation_predictions[i] == None:
                self.validation_predictions[i] = [None for _ in range(batch_input_size)]
        print(f'Predictions {self.validation_predictions}')
        #if len(args) ==5:
        #    breakpoint()
        return self.validation_predictions
    def set_gradient_accumulation_interval(self, task_id, interval):
        self.gradient_accumulation_steps[task_id] = interval
    def train_on_batch(self,*args):
        """Train model on given batch.
        This method calls train_op using features and y (labels).
        Args:
            features: batch of InputFeatures
            y: batch of labels (class id)
        Returns:
            dict with loss for each task
        """
        assert len(args) == 2*self.n_tasks, breakpoint() # need to get exact number of x andy
        ids_to_iterate = [k for k in range(self.n_tasks) if len(args[k])>0]
        assert len(ids_to_iterate) == 1, breakpoint() # 1 batch 1 task
        task_id = ids_to_iterate[0]
        print('train on batch')
        _input,batch_size = self._make_input(task_features=args[task_id],task_id=task_id,
                                labels=args[task_id+self.n_tasks])
        if len(args) ==5:
            breakpoint()
        if self.prev_id is None:
            self.prev_id = task_id
        elif self.prev_id != task_id and not self.printed:
            log.info('Seen samples from different tasks')
            self.printed = True
        loss, logits = self.model(
                task_id=task_id, name=self.tasks_type[task_id], **_input)

        loss = loss / self.gradient_accumulation_steps[task_id]
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_norm)

        if (self.steps_taken + 1) % self.gradient_accumulation_steps[task_id] == 0 or (
            self.steps_per_epoch is not None and (self.steps_taken + 1) % self.steps_per_epoch == 0):
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()  # Update learning rate schedule
            self.optimizer.zero_grad()
        self.train_losses[task_id] = loss.item()
        self.steps_taken += 1
        return {"losses": self.train_losses}
