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
import torch
import os
from .optimization import BERTAdam
from .modeling import BertForMultiTask, BertConfig
from transformers.data.processors.utils import InputFeatures

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
        config: path to pal Bert configuration file,
        pretrained_bert: path of the pretrained bert embeddings
        freeze_embeddings: set True if bert embeddings are to be freezed
        learning_rate: learning rate for the model
        steps_per_epoch: number of steps per epoch,
        num_train_epochs: number of epochs,
        warmup_proportion: warmup proportion for the optimizer,
        gradient_accumulation_steps: number of gradient accumulation steps,
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
                 config: str = None,
                 pretrained_bert: str = None,
                 freeze_embeddings: bool = False,
                 learning_rate: int = 2e-5,
                 steps_per_epoch: int = 2400,
                 num_train_epochs: int = 3,
                 warmup_proportion: int = 0.1,
                 gradient_accumulation_steps: int = 1,
                 in_distribution: Optional[Union[Dict[str, int], Dict[str, List[str]]]] = None,
                 in_y_distribution: Optional[Union[Dict[str, int], Dict[str, List[str]]]] = None,
                 *args,
                 **kwargs) -> None:

        self.task_names = list(tasks.keys())
        self.tasks_num_classes = [tasks[task]["n_classes"] for task in tasks]
        self.losses_template = [[] for task in self.task_names]
        self.train_losses = self.losses_template.copy()
        self.validation_predictions = self.losses_template.copy()
        self.validation_steps = 0
        self.training_steps = 0
        self.config = config
        self.pretrained_bert = pretrained_bert
        self.freeze_embeddings = freeze_embeddings
        self.learning_rate = learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.in_distribution = in_distribution
        self.in_y_distribution = in_y_distribution
        super().__init__(*args, **kwargs)

    def load(self, fname: Optional[str] = None, *args, **kwargs) -> None:
        if self.config and os.path.exists(self.config):
            self.config = BertConfig.from_json_file(self.config)
            self.config.num_tasks = len(self.task_names)
            self.model = BertForMultiTask(
                self.config, self.tasks_num_classes)
        else:
            raise ValueError("Config File does not exist at", self.config)

        if fname is not None:
            self.load_path = fname

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
                    update[n] = partial[n]
            self.model.bert.load_state_dict(update)
            log.info("Bert Model Weights Loaded.")
        else:
            raise ConfigError("No pre-trained BERT model is given.")

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
        self.model.to(self.device)
        no_decay = ["bias", "gamma", "beta"]
        base = ["attn"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in base)
                ],
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(nd in n for nd in base)
                ],
                "weight_decay_rate": 0.0,
            },
        ]
        self.optimizer = BERTAdam(
            optimizer_parameters,
            lr=self.learning_rate,
            warmup=self.warmup_proportion,
            t_total=self.steps_per_epoch * self.num_train_epochs
        )

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

    def __call__(self, *args):
        """Make prediction for given features (texts).
        Args:
            features: batch of InputFeatures for all tasks
        Returns:
            predicted classes or probabilities of each class
        """
        if self.validation_steps >= (self.steps_per_epoch / self.gradient_accumulation_steps):
            self.validation_steps = 0
            self.validation_predictions = self.losses_template.copy()
        self.validation_steps += 1
        n_in = sum([inp for inp in self.in_distribution.values()])
        task_id = args[0]
        features = args[1:]
        args_in = features[:n_in]
        in_by_tasks = self._distribute_arguments_by_tasks(
            args_in, {}, self.task_names, "in")

        task_features = in_by_tasks[self.task_names[task_id]]

        input_ids = []
        attention_masks = []
        token_type_ids = []
        for input_feature in task_features[0]:
            input_ids.append(input_feature.input_ids.numpy()[0])
            attention_masks.append(
                input_feature.attention_mask.numpy()[0])
            token_type_ids.append(
                input_feature.token_type_ids.numpy()[0])
        logits = self.model(
            torch.tensor(input_ids, dtype=torch.long).to(self.device),
            torch.tensor(attention_masks, dtype=torch.long).to(
                self.device),
            torch.tensor(token_type_ids, dtype=torch.long).to(
                self.device),
            task_id,
            self.task_names[task_id],
        )
        logits = logits.detach().cpu().numpy()
        self.validation_predictions[task_id] = logits.tolist()

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
        if self.training_steps >= (self.steps_per_epoch / self.gradient_accumulation_steps):
            self.training_steps = 0
            self.train_losses = self.losses_template.copy()
        self.training_steps += 1
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

        input_ids = []
        attention_masks = []
        token_type_ids = []

        for input_feature in task_features[0]:
            input_ids.append(input_feature.input_ids.numpy()[0])
            attention_masks.append(
                input_feature.attention_mask.numpy()[0])
            token_type_ids.append(
                input_feature.token_type_ids.numpy()[0])
        loss, logits = self.model(
            torch.tensor(input_ids, dtype=torch.long).to(self.device),
            torch.tensor(attention_masks, dtype=torch.long).to(
                self.device),
            torch.tensor(token_type_ids, dtype=torch.long).to(
                self.device),
            task_id,
            self.task_names[task_id],
            torch.tensor(task_labels[0], dtype=torch.long).to(self.device),
        )
        if self.gradient_accumulation_steps > 1:
            loss = loss/self.gradient_accumulation_steps
        loss.backward()
        self.optimizer.step()
        self.model.zero_grad()
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
