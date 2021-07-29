# Copyright 2021 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from logging import getLogger
from typing import List, Dict, Union, Optional
from pathlib import Path

import numpy as np
from overrides import overrides
import torch
import os

from .modeling import BertForMultiTask, BertConfig

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel

log = getLogger(__name__)


@register('multitask_pal_bert')
class MultiTaskPalBert(TorchModel):
    """ Multi-Task Bert Based Model 
    Args:
        tasks: Dict of task names along with the labels for each task,
        pretrained_bert: path of the pretrained bert embeddings
        freeze_embeddings: set True if bert embeddings are to be freezed,
        optimizer: optimizer name defaults to AdamW,
        optimizer_parameters: optimizer parameters,
        lr_scheduler: name of the lr scheduler,
        lr_scheduler_paramters: lr scheduler parameters for the scheduler,
        gradient_accumulation_steps: number of gradient accumulation steps,
        clip_norm: normalization: value for gradient clipping,
        one_hot_labels: set to true if using one hot labels,
        multilabel: set true for multilabel class,
        return_probas: set true to return prediction probas,
        in_distribution: in_distribution: The distribution of variables listed in the ``"in"`` config parameter between tasks. 
            ``in_distribution`` can be ``None`` if only 1 task is called. In that case all variables
            listed in ``"in"`` are arguments of 1 task. 
            ``in_distribution`` can be a dictionary of ``int``. If that is the case, then keys of ``in_distribution``
            are task names and values are numbers of variables from ``"in"`` parameter of config which are inputs of
            corresponding task. The variables in ``"in"`` parameter have to be in the same order the tasks are listed
            in ``in_distribution``.
        in_y_distribution: The same as ``in_distribution`` for ``"in_y"`` config parameter.,
    """

    def __init__(self,
                 tasks: Dict[str, Dict],
                 pretrained_bert: str = None,
                 freeze_embeddings: bool = False,
                 optimizer: str = "AdamW",
                 optimizer_parameters: dict = {"lr": 2e-5},
                 lr_scheduler: Optional[str] = None,
                 lr_scheduler_parameters: dict = {},
                 gradient_accumulation_steps: int = 1,
                 clip_norm: Optional[float] = None,
                 one_hot_labels: bool = False,
                 multilabel: bool = False,
                 return_probas: bool = False,
                 in_distribution: Optional[Union[Dict[str, int], Dict[str, List[str]]]] = None,
                 in_y_distribution: Optional[Union[Dict[str, int], Dict[str, List[str]]]] = None,
                 *args,
                 **kwargs) -> None:
        path_to_current_file = os.path.realpath(__file__)
        current_directory = os.path.split(path_to_current_file)[0]
        self.config = os.path.join(
            current_directory, "configs/pals_config.json")
        self.return_probas = return_probas
        self.one_hot_labels = one_hot_labels
        self.multilabel = multilabel
        self.clip_norm = clip_norm
        self.task_names = list(tasks.keys())
        self.tasks_num_classes = [tasks[task]["n_classes"] for task in tasks]
        self.tasks_type = ["regression" if num_classes ==
                           1 else "classification" for num_classes in self.tasks_num_classes]
        self.train_losses = [[] for task in self.task_names]
        self.pretrained_bert = pretrained_bert
        self.freeze_embeddings = freeze_embeddings
        self.optimizer_name = optimizer
        self.optimizer_parameters = optimizer_parameters
        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_parameters = lr_scheduler_parameters
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.in_distribution = in_distribution
        self.in_y_distribution = in_y_distribution

        if self.multilabel and not self.one_hot_labels:
            raise RuntimeError(
                'Use one-hot encoded labels for multilabel classification!')

        if self.multilabel and not self.return_probas:
            raise RuntimeError(
                'Set return_probas to True for multilabel classification!')

        super().__init__(
            optimizer_parameters=self.optimizer_parameters,
            lr_scheduler=self.lr_scheduler_name,
            lr_scheduler_parameters=self.lr_scheduler_parameters,
            **kwargs
        )

    @overrides
    def init_from_opt(self) -> None:
        """Initialize from scratch `self.model` with the architecture built in  `model_func (MultitaskBert)` method of this class
            along with `self.optimizer` as `self.optimizer_name` from `torch.optim` and parameters
            `self.optimizer_parameters`, optionally initialize `self.lr_scheduler` as `self.lr_scheduler_name` from
            `torch.optim.lr_scheduler` and parameters `self.lr_scheduler_parameters`
        """
        if self.config and os.path.exists(self.config):
            self.bert_config = BertConfig.from_json_file(self.config)
            self.bert_config.num_tasks = len(self.task_names)
            self.model = BertForMultiTask(
                self.bert_config, self.tasks_num_classes)
            self.model.to(self.device)
        else:
            raise ValueError("Config File does not exist at", self.config)

        if self.pretrained_bert:
            partial = torch.load(self.pretrained_bert, map_location="cpu")
            model_dict = self.model.bert.state_dict()
            update = {}
            for n, p in model_dict.items():
                if "aug" in n or "mult" in n:
                    update[n] = p
                    if "pooler.mult" in n and "bias" in n:
                        update[n] = partial["pooler.dense.bias"]
                    if "pooler.mult" in n and "weight" in n:
                        update[n] = partial["pooler.dense.weight"]
                else:
                    for val in [n, 'bert.'+n, 'cls.'+n]:
                        if val in partial:
                            update[n] = partial[val]
            self.model.bert.load_state_dict(update)
            log.info("Bert Model Weights Loaded.")
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        no_decay = ["bias", "gamma", "beta"]
        base = ["attn"]
        model_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in base)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in base)
                ],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = getattr(torch.optim, self.optimizer_name)(
            model_parameters, **self.optimizer_parameters)

        if self.lr_scheduler_name:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)

    @overrides
    def load(self, fname: Optional[str] = None) -> None:
        if fname is not None:
            self.load_path = fname

        if self.load_path:
            log.info(f"Load path {self.load_path} is given.")
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            weights_path = Path(self.load_path.resolve())
            weights_path = weights_path.with_suffix(f".pth.tar")
            if weights_path.exists():
                log.info(f"Load path {weights_path} exists.")
                log.info(
                    f"Initializing `{self.__class__.__name__}` from saved.")

                # firstly, initialize with random weights and previously saved parameters
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

    def __call__(self, *args):
        """Make prediction for given features (texts).
        Args:
            features: batch of InputFeatures for all tasks
        Returns:
            predicted classes or probabilities of each class
        """
        self.validation_predictions = []
        n_in = sum([inp for inp in self.in_distribution.values()])

        features = args[1:]
        args_in = features[:n_in]
        in_by_tasks = self._distribute_arguments_by_tasks(
            args_in, {}, self.task_names, "in")

        for task_id in range(len(self.task_names)):
            task_features = in_by_tasks[self.task_names[task_id]]

            _input = {}
            for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
                _input[elem] = [getattr(f, elem) for f in task_features[0]]

            for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
                _input[elem] = torch.cat(_input[elem], dim=0).to(self.device)

            with torch.no_grad():
                tokenized = {key: value for (key, value) in _input.items(
                ) if key in self.model.forward.__code__.co_varnames}

                logits = self.model(
                    task_id=task_id,
                    name=self.tasks_type[task_id],
                    **tokenized
                )

            if self.return_probas:
                if not self.multilabel:
                    pred = torch.nn.functional.softmax(logits, dim=-1)
                else:
                    pred = torch.nn.functional.sigmoid(logits)
                pred = pred.detach().cpu().numpy()
            elif self.tasks_num_classes[task_id] > 1:
                logits = logits.detach().cpu().numpy()
                pred = np.argmax(logits, axis=1)
            else:  # regression
                pred = logits.squeeze(-1).detach().cpu().numpy()
            self.validation_predictions.append(pred)
        return self.validation_predictions

    def train_on_batch(self, *args):
        """Train model on given batch.
        This method calls train_op using features and y (labels).
        Args:
            features: batch of InputFeatures
            y: batch of labels (class id)
        Returns:
            dict with loss for each task
        """
        n_in = sum([inp for inp in self.in_distribution.values()])
        task_id = args[0]
        features = args[1:]
        args_in, args_in_y = features[:n_in], features[n_in:]
        in_by_tasks = self._distribute_arguments_by_tasks(
            args_in, {}, self.task_names, "in")
        in_y_by_tasks = self._distribute_arguments_by_tasks(
            args_in_y, {}, self.task_names, "in_y")

        task_features = in_by_tasks[self.task_names[task_id]]
        task_labels = in_y_by_tasks[self.task_names[task_id]]

        _input = {}
        for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
            _input[elem] = [getattr(f, elem) for f in task_features[0]]

        for elem in ['input_ids', 'attention_mask', 'token_type_ids']:
            _input[elem] = torch.cat(_input[elem], dim=0).to(self.device)

        if self.tasks_type[task_id] == "regression":
            _input['labels'] = torch.tensor(
                np.array(task_labels[0], dtype=float), dtype=torch.float32).to(self.device)
        else:
            _input['labels'] = torch.from_numpy(
                np.array(task_labels[0])).to(self.device)
        tokenized = {key: value for (key, value) in _input.items(
        ) if key in self.model.forward.__code__.co_varnames}

        loss, logits = self.model(
            task_id=task_id,
            name=self.tasks_type[task_id],
            **tokenized
        )
        if self.gradient_accumulation_steps > 1:
            loss = loss/self.gradient_accumulation_steps
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_norm)

        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()  # Update learning rate schedule
        self.optimizer.zero_grad()
        self.train_losses[task_id] = loss.item()

        return {"losses": self.train_losses}

    def _distribute_arguments_by_tasks(self, args, kwargs, task_names, what_to_distribute, in_distribution=None):

        if args and kwargs:
            raise ValueError("You may use args or kwargs but not both")

        if what_to_distribute == "in":
            if in_distribution is not None:
                distribution = in_distribution
            else:
                distribution = self.in_distribution
        elif what_to_distribute == "in_y":
            if in_distribution is not None:
                raise ValueError(
                    f"If parameter `what_to_distribute` is 'in_y', parameter `in_distribution` has to be `None`. "
                    f"in_distribution = {in_distribution}")
            distribution = self.in_y_distribution
        else:
            raise ValueError(
                f"`what_to_distribute` can be 'in' or 'in_y', {repr(what_to_distribute)} is given")

        if distribution is None:
            if len(task_names) != 1:
                raise ValueError(f"If no `{what_to_distribute}_distribution` is not provided there have to be only 1"
                                 "task for inference")
            return {task_names[0]: list(kwargs.values()) if kwargs else list(args)}

        if all([isinstance(task_distr, int) for task_distr in distribution.values()]):
            ints = True
        elif all([isinstance(task_distr, list) for task_distr in distribution.values()]):
            ints = False
        else:
            raise ConfigError(
                f"Values of `{what_to_distribute}_distribution` attribute of `MultiTaskBert` have to be "
                f"either `int` or `list` not both. "
                f"{what_to_distribute}_distribution = {distribution}")

        args_by_task = {}

        flattened = []
        for task_name in task_names:
            if isinstance(task_name, str):
                flattened.append(task_name)
            else:
                flattened.extend(task_name)
        task_names = flattened

        if args and not ints:
            ints = True
            distribution = {task_name: len(
                in_distr) for task_name, in_distr in distribution.items()}
        if ints:
            if kwargs:
                values = list(kwargs.values())
            else:
                values = args
            n_distributed = sum([n_args for n_args in distribution.values()])
            if len(values) != n_distributed:
                raise ConfigError(
                    f"The number of '{what_to_distribute}' arguments of MultitaskBert does not match "
                    f"the number of distributed params according to '{what_to_distribute}_distribution' parameter. "
                    f"{len(values)} parameters are in '{what_to_distribute}' and {n_distributed} parameters are "
                    f"required '{what_to_distribute}_distribution'. "
                    f"{what_to_distribute}_distribution = {distribution}")
            values_taken = 0
            for task_name in task_names:
                args_by_task[task_name] = {}
                n_args = distribution[task_name]
                args_by_task[task_name] = [values[i]
                                           for i in range(values_taken, values_taken + n_args)]
                values_taken += n_args

        else:
            assert kwargs
            arg_names_used = []
            for task_name in task_names:
                in_distr = distribution[task_name]
                args_by_task[task_name] = {}
                args_by_task[task_name] = [kwargs[arg_name]
                                           for arg_name in in_distr]
                arg_names_used += in_distr
            set_used = set(arg_names_used)
            set_all = set(kwargs.keys())
            if set_used != set_all:
                raise ConfigError(
                    f"There are unused '{what_to_distribute}' parameters {set_all - set_used}")
        return args_by_task
