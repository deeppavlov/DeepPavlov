from logging import getLogger
from typing import Dict, Optional
from pathlib import Path
import numpy as np
from overrides import overrides
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from collections.abc import Iterable
from transformers import AutoConfig, AutoModel

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.models.torch_bert.torch_transformers_sequence_tagger import token_from_subtoken
from deeppavlov.models.torch_bert.torch_transformers_sequence_tagger import token_labels_to_subtoken_labels

log = getLogger(__name__)

prev_input = None


class FixSizeOrderedDict(OrderedDict):
    def __init__(self, *args, max=0, **kwargs):
        self._max = max
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._max > 0:
            if len(self) > self._max:
                self.popitem(False)


def we_transform_input(name):
    return name in ['sequence_labeling', 'multiple_choice']


class BertForMultiTask(nn.Module):
    """
    BERT model for multiple choice,sequence labeling, ner, classification or regression
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
    task_num_classes
    task_types
    backbone_model - na
    """

    def __init__(self, tasks_num_classes, multilabel, task_types,
                 backbone_model='bert_base_uncased', dropout=None,
                 max_seq_len=320):

        super(BertForMultiTask, self).__init__()
        config = AutoConfig.from_pretrained(
            backbone_model, output_hidden_states=True, output_attentions=True)
        self.bert = AutoModel.from_pretrained(pretrained_model_name_or_path=backbone_model,
                                              config=config)
        self.classes = tasks_num_classes  # classes for every task
        self.multilabel = multilabel
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        elif hasattr(config, 'hidden_dropout_prob'):
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
        else:
            self.dropout = nn.Dropout(0)
        self.max_seq_len = max_seq_len
        self.activation = nn.Tanh()
        self.task_types = task_types
        OUT_DIM = config.hidden_size
        self.bert.final_classifier = nn.ModuleList(
            [
                nn.Linear(OUT_DIM, num_labels) if self.task_types[i] not in ['multiple_choice', 'regression']
                else nn.Linear(OUT_DIM, 1) for i, num_labels in enumerate(self.classes)
            ]
        )
        self.bert.pooler = nn.Linear(OUT_DIM, OUT_DIM)

    def get_logits(self, task_id, input_ids, attention_mask, token_type_ids):
        name = self.task_types[task_id]
        outputs = None
        if we_transform_input(name):
            input_ids = input_ids.view(-1, input_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        if token_type_ids is None:
            outputs = self.bert(input_ids=input_ids.long(),
                                attention_mask=attention_mask.long())
        else:
            outputs = self.bert(input_ids=input_ids.long(),
                                token_type_ids=token_type_ids.long(),
                                attention_mask=attention_mask.long())
        if name == 'sequence_labeling':
            return outputs.last_hidden_state
        else:
            return outputs.last_hidden_state[:, 0]

    def predict_on_top(self, task_id, last_hidden_state, labels=None):
        name = self.task_types[task_id]
        if name == 'sequence_labeling':
            #  last hidden state is all token tensor
            final_output = self.dropout(last_hidden_state)
            logits = self.bert.final_classifier[task_id](final_output)
            if labels is not None:
                active_logits = logits.view(-1, self.classes[task_id])
                if not self.multilabel[task_id]:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(active_logits, labels.view(-1))
                elif self.multilabel[task_id]:
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(active_logits, labels)
                return loss, logits
            else:
                return logits
        elif name in ['classification', 'regression', 'multiple_choice']:
            #  last hidden state is a first token tensor
            pooled_output = self.bert.pooler(last_hidden_state)
            pooled_output = self.activation(pooled_output)
            pooled_output = self.dropout(pooled_output)
            logits = self.bert.final_classifier[task_id](pooled_output)
            if name == 'multiple_choice':
                logits = logits.view((-1, self.classes[task_id]))
                if labels is not None:
                    l1, l2 = len(logits), len(labels)
                    assert len(logits) == len(
                        labels), f'Len of logits {l1} and labels {l2} not match'
            if labels is not None:
                if name != "regression":
                    if not self.multilabel[task_id]:
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(logits, labels.view(-1))
                    elif self.multilabel[task_id]:
                        loss_fct = BCEWithLogitsLoss()
                        loss = loss_fct(logits, labels)
                    return loss, logits
                elif name == "regression":
                    loss_fct = MSELoss()
                    logits = logits.cpu()
                    labels = labels.cpu()
                    loss = loss_fct(logits, labels.unsqueeze(1))
                    return loss, logits
            else:
                return logits
        else:
            raise Exception(f'Unsupported name {name}')

    def forward(
            self,
            task_id,
            input_ids,
            attention_mask,
            token_type_ids,
            labels=None
    ):
        last_hidden_state = self.get_logits(task_id, input_ids, attention_mask, token_type_ids)
        return self.predict_on_top(task_id, last_hidden_state, labels)


@register('multitask_bert')
class TorchMultiTaskBert(TorchModel):
    """
    Multi-Task transformer-agnostic model
    Args:
        tasks: Dict of task names along with the labels for each task,
        max_seq_len(int): maximum length of the input token sequence.
        optimizer(str): optimizer name defaults to AdamW,
        optimizer_parameters(dict): optimizer parameters,
        lr_scheduler(str): name of the lr scheduler,if it is used
        lr_scheduler_parameters(dict): lr scheduler parameters for the scheduler, if the scheduler is used
        gradient_accumulation_steps(default:1): number of gradient accumulation steps,
        steps_per_epoch(int): number of steps taken per epoch. Specify if gradient_accumulation_steps > 1
        backbone_model(str): name of HuggingFace.Transformers backbone model. Default: 'bert-base-cased'
        clip_norm(float): normalization: value for gradient clipping. Specify only if gradient clipping is used
        one_hot_labels(default: False): set to true if using one hot labels,
        multilabel(default: False): set to true for multilabel classification,
        return_probas(default: False): set true to return prediction probabilities,
        freeze_embeddings(default: False): set true to freeze BERT embeddings
        dropout(default: None): dropout for the final model layer.
        cache_size(default:10): cache size for the last predicts that we use
        If not set, defaults to the parameter hidden_dropout_prob of original model
    """

    def __init__(
            self,
            tasks: Dict[str, Dict],
            max_seq_len: str = 320,
            optimizer: str = "AdamW",
            optimizer_parameters: dict = {"lr": 2e-5},
            lr_scheduler: Optional[str] = None,
            lr_scheduler_parameters: dict = {},
            gradient_accumulation_steps: Optional[int] = 1,
            steps_per_epoch: Optional[int] = None,
            backbone_model: str = "bert-base-cased",
            clip_norm: Optional[float] = None,
            one_hot_labels: bool = False,
            return_probas: bool = False,
            freeze_embeddings: bool = False,
            dropout: Optional[float] = None,
            cache_size: int = 3,
            *args,
            **kwargs,
    ) -> None:
        path_to_current_file = os.path.realpath(__file__)
        self.return_probas = return_probas
        self.one_hot_labels = one_hot_labels
        self.clip_norm = clip_norm
        self.task_names = list(tasks.keys())
        self.task_types = []
        self.backbone_model = backbone_model
        self.max_seq_len = max_seq_len
        self.tasks_num_classes = []
        self.task_names = []
        self.multilabel = []
        for task in tasks:
            self.task_names.append(task)
            self.tasks_num_classes.append(tasks[task]['options'])
            self.task_types.append(tasks[task]['type'])
            self.multilabel.append(tasks[task].get('multilabel', False))
        if self.return_probas and 'sequence_labeling' in self.task_types:
            log.warning(
                f'Return_probas for sequence_labeling not supported yet. Returning ids for this task')
        self.n_tasks = len(tasks)
        self.train_losses = [[] for task in self.task_names]
        self.optimizer_name = optimizer
        self.optimizer_parameters = optimizer_parameters
        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_parameters = lr_scheduler_parameters
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.steps_per_epoch = steps_per_epoch
        self.steps_taken = 0
        self.prev_id = None
        self.printed = False
        self.freeze_embeddings = freeze_embeddings
        self.dropout = dropout
        self.cache = FixSizeOrderedDict(max=cache_size)

        super().__init__(
            optimizer_parameters=self.optimizer_parameters,
            lr_scheduler=self.lr_scheduler_name,
            lr_scheduler_parameters=self.lr_scheduler_parameters,
            **kwargs,
        )

    @overrides
    def init_from_opt(self) -> None:
        """
        Initialize from scratch `self.model` with the architecture built
        in `model_func (MultitaskBert)` method of this class along with
        `self.optimizer` as `self.optimizer_name` from `torch.optim` and
        parameters `self.optimizer_parameters`, optionally initialize
        `self.lr_scheduler` as `self.lr_scheduler_name` from
        `torch.optim.lr_scheduler` and parameters `self.lr_scheduler_parameters`
        """

        self.model = BertForMultiTask(
            backbone_model=self.backbone_model,
            tasks_num_classes=self.tasks_num_classes,
            multilabel=self.multilabel,
            task_types=self.task_types,
            dropout=self.dropout)
        self.model = self.model.to(self.device)
        no_decay = ["bias", "gamma", "beta"]
        base = ["attn"]

        def get_non_decay_params(model): return [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
               and not any(nd in n for nd in base)
        ]

        def get_decay_params(model): return [
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

    @overrides
    def load(self, fname: Optional[str] = None) -> None:
        """
        Loads weights.
        """
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

        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

    def _make_input(self, task_features, task_id, labels=None):
        batch_input_size = None
        if len(task_features) == 1 and isinstance(task_features, list):
            task_features = task_features[0]

        if isinstance(labels, Iterable) and all([k is None for k in labels]):
            labels = None
        _input = {}
        element_list = ["input_ids", "attention_mask", "token_type_ids"]
        for elem in element_list:
            if elem in task_features:
                _input[elem] = task_features[elem]
                batch_input_size = _input[elem].shape[0]
            elif hasattr(task_features, elem):
                _input[elem] = getattr(task_features, elem)
                batch_input_size = _input[elem].shape[0]
            if elem in _input:
                if we_transform_input(self.task_types[task_id]):
                    _input[elem] = _input[elem].view(
                        (-1, _input[elem].size(-1)))

        if labels is not None:
            if self.task_types[task_id] == "regression":
                _input["labels"] = torch.tensor(
                    np.array(labels, dtype=float), dtype=torch.float32
                )
            elif self.task_types[task_id] == 'multiple_choice':
                labels = torch.Tensor(labels).long()
                _input['labels'] = labels
            elif self.task_types[task_id] == 'sequence_labeling':
                subtoken_labels = [token_labels_to_subtoken_labels(y_el, y_mask, input_mask)
                                   for y_el, y_mask, input_mask in zip(labels, _input['token_type_ids'].numpy(),
                                                                       _input['attention_mask'].numpy())]
                _input['labels'] = torch.from_numpy(
                    np.array(subtoken_labels)).to(torch.int64)
            else:
                if not self.multilabel[task_id]:
                    _input["labels"] = torch.from_numpy(np.array(labels))
                elif self.multilabel[task_id]:
                    # We assume that labels already are one hot encoded
                    num_classes = self.tasks_num_classes[task_id]
                    _input['labels'] = torch.zeros((len(labels), num_classes))
                    for i in range(len(labels)):
                        for label_ind in labels[i]:
                            _input['labels'][i][label_ind] = 1
            element_list = element_list + ['labels']
        for elem in element_list:
            if elem not in _input:
                _input[elem] = None
            else:
                _input[elem] = _input[elem].to(self.device)
        if 'labels' in _input and self.task_types[task_id] != 'multiple_choice':
            error_msg = f'Len of labels {len(_input["labels"])} does not match len of ids {len(_input["input_ids"])}'
            assert len(_input['labels']) == len(_input['input_ids']), error_msg
        return _input, batch_input_size

    def __call__(self, *args):
        """Make prediction for given features (texts).
        Args:
            features: batch of InputFeatures for all tasks
        Returns:
            predicted classes or probabilities of each class
        """
        # IMPROVE ARGS CHECKING AFTER DEBUG
        log.debug(f'Calling {args}')
        self.validation_predictions = [None for _ in range(len(args))]
        for task_id in range(len(self.task_names)):
            if len(args[task_id]):
                _input, batch_input_size = self._make_input(
                    task_features=args[task_id], task_id=task_id)

                assert 'input_ids' in _input, f'No input_ids in _input {_input}'
                cache_key = (we_transform_input(self.task_names[task_id]),
                             str(args[task_id]))
                if cache_key in self.cache:
                    last_hidden_state = self.cache[cache_key]
                else:
                    with torch.no_grad():
                        last_hidden_state = self.model.get_logits(task_id, **_input)
                        self.cache[cache_key] = last_hidden_state
                with torch.no_grad():
                    logits = self.model.predict_on_top(task_id, last_hidden_state)
                if self.task_types[task_id] == 'sequence_labeling':
                    y_mask = _input['token_type_ids'].cpu()
                    logits = token_from_subtoken(logits.cpu(), y_mask)
                    predicted_ids = torch.argmax(logits, dim=-1).int().tolist()
                    seq_lengths = torch.sum(y_mask, dim=1).int().tolist()
                    pred = [prediction[:max_seq_len] for max_seq_len,
                                                         prediction in zip(seq_lengths, predicted_ids)]
                elif self.task_types[task_id] == 'regression':
                    pred = logits[:, 0]
                else:
                    if self.multilabel[task_id]:
                        probs = torch.sigmoid(logits)
                        if self.return_probas:
                            pred = probs
                        else:
                            pred = [torch.where(k > 0.5)[0].cpu().tolist() for k in probs]
                    else:
                        if self.return_probas:
                            pred = torch.softmax(logits, dim=-1)
                        else:
                            pred = torch.argmax(logits, dim=1)
                    if not isinstance(pred, list):
                        pred = pred.tolist()
                self.validation_predictions[task_id] = pred
        log.debug(f'Predictions {self.validation_predictions}')
        if len(args) == 1:
            return self.validation_predictions[0]
        for i in range(len(self.validation_predictions)):
            if self.validation_predictions[i] is None:
                self.validation_predictions[i] = [
                    None for _ in range(batch_input_size)]
        return self.validation_predictions

    def set_gradient_accumulation_interval(self, task_id, interval):
        self.gradient_accumulation_steps[task_id] = interval

    def train_on_batch(self, *args):
        """Train model on given batch.
        This method calls train_op using features and y (labels).
        Args:
            features: batch of InputFeatures
            y: batch of labels (class id)
        Returns:
            dict with loss for each task
        """
        log.debug(f'Training for {args}')
        error_msg = f'Len of arguments {len(args)} is WRONG. ' \
                    f'Correct is {2 * self.n_tasks} as n_tasks is {self.n_tasks}'
        assert len(args) == 2 * self.n_tasks, error_msg
        ids_to_iterate = [k for k in range(self.n_tasks) if len(args[k]) > 0]
        assert len(
            ids_to_iterate) == 1, 'Samples from more than 1 task in train_on_batch'
        task_id = ids_to_iterate[0]
        _input, batch_size = self._make_input(task_features=args[task_id], task_id=task_id,
                                              labels=args[task_id + self.n_tasks])
        assert _input != {}, 'Empty input!'

        if self.prev_id is None:
            self.prev_id = task_id
        elif self.prev_id != task_id and not self.printed:
            log.info('Seen samples from different tasks')
            self.printed = True
        if 'token_type_ids' not in _input:
            _input['token_type_ids'] = None
        loss, logits = self.model(task_id=task_id, **_input)
        if self.is_data_parallel:
            loss = loss.mean()
        loss = loss / self.gradient_accumulation_steps
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_norm)

        if (self.steps_taken + 1) % self.gradient_accumulation_steps == 0 or (
                self.steps_per_epoch is not None and (self.steps_taken + 1) % self.steps_per_epoch == 0):
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()  # Update learning rate schedule
            self.optimizer.zero_grad()
        self.train_losses[task_id] = loss.item()
        self.steps_taken += 1
        return {"losses": self.train_losses}
