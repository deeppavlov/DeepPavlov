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
from numpy.core.fromnumeric import argsort
from overrides import overrides
import torch
import os
from transformers.tokenization_utils_base import BatchEncoding

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.models.multitask_pal_bert.modeling import BertForMultiTask, BertConfig
from deeppavlov.models.torch_bert.torch_transformers_sequence_tagger import token_from_subtoken, token_labels_to_subtoken_labels
from deeppavlov.models.torch_bert.torch_transformers_squad import softmax_mask

log = getLogger(__name__)

VERY_LOW_VALUE = -99999

def transform(list_of_features):
    ans = {}
    for feature in list_of_features:
        for key in feature.__dict__:
            if feature.__dict__[key] is not None:
                if key not in ans:
                    ans[key] = feature.__dict__[key]
                else:
                    ans[key] = torch.cat((ans[key],feature.__dict__[key]),0)
    return ans

@register('multitask_pal_bert')
class MultiTaskPalBert(TorchModel):
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
        optimizer: str = "AdamW",
        optimizer_parameters: dict = {"lr": 2e-5,
        },
        lr_scheduler: Optional[str] = None,
        lr_scheduler_parameters: dict = {},
        gradient_accumulation_steps: Optional[int] = 1,
        steps_per_epoch: Optional[int] = None,
        clip_norm: Optional[float] = None,
        one_hot_labels: bool = False,
        multilabel: bool = False,
        return_probas: bool = False,
        config_name: str =  "configs/pals_config.json",
        in_distribution: Optional[Union[Dict[str, int], Dict[str, List[str]]]] = None,
        in_y_distribution: Optional[Union[Dict[str, int], Dict[str, List[str]]]] = None,
        *args,
        **kwargs,
    ) -> None:
        path_to_current_file = os.path.realpath(__file__)
        current_directory = os.path.split(path_to_current_file)[0]
        self.config = os.path.join(
            current_directory,config_name)
        self.return_probas = return_probas
        self.one_hot_labels = one_hot_labels
        self.clip_norm = clip_norm
        self.task_names = list(tasks.keys())
        self.tasks_num_classes = []
        self.tasks_type = []
        for task in tasks:
            if 'question_answering' not in tasks[task]:
                assert 'n_choices' in tasks[task] or 'n_classes' in tasks[task], 'Provide n_classes or n_choices'
            n_classes = tasks[task].get("n_classes", 0)
            n_choices = tasks[task].get("n_choices", 0)
            is_question_answering = 'question_answering' in tasks[task] and n_classes == 0 and n_choices == 0
            if is_question_answering:
                self.tasks_type.append('question_answering')
                self.tasks_num_classes.append(2)                
            elif n_choices > 0:
                assert isinstance(n_choices, int) and n_choices > 0
                self.tasks_type.append('sequence_labeling')
                self.tasks_num_classes.append(n_choices)
            elif n_classes == 1:
                self.tasks_type.append("regression")
                self.tasks_num_classes.append(n_classes)
            else:
                assert isinstance(n_classes, int) and n_classes > 0
                self.tasks_type.append("classification")
                self.tasks_num_classes.append(n_classes)
        self.train_losses = [[] for task in self.task_names]
        self.pretrained_bert = pretrained_bert
        self.freeze_embeddings = freeze_embeddings
        self.optimizer_name = optimizer
        self.optimizer_parameters = optimizer_parameters
        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_parameters = lr_scheduler_parameters
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.steps_per_epoch = steps_per_epoch
        self.in_distribution = in_distribution
        self.in_y_distribution = in_y_distribution
        self.steps_taken = 0
        if multilabel:
            log.warning('Multilabel classification is not supported')

        if self.gradient_accumulation_steps > 1 and not self.steps_per_epoch:
            raise RuntimeError(
                "Provide steps per epoch when using gradient accumulation"
            )

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
        if self.config and os.path.exists(self.config):
            self.bert_config = BertConfig.from_json_file(self.config)
            self.bert_config.num_tasks = len(self.task_names)
            self.model = BertForMultiTask(
                self.bert_config, self.tasks_num_classes, self.tasks_type)
            self.model.to(self.device)
        else:
            raise ValueError("Config File does not exist at", self.config)

        if self.pretrained_bert:
            partial = torch.load(self.pretrained_bert, map_location="cpu")
            # print(partial.keys())
            model_dict = self.model.bert.state_dict()
            # print(model_dict.keys())
            update = {}
            for n, p in model_dict.items():
                if "aug" in n or "mult" in n:
                    update[n] = p
                    if "pooler.mult" in n and "bias" in n:
                        update[n] = partial["pooler.dense.bias"]
                    if "pooler.mult" in n and "weight" in n:
                        update[n] = partial["pooler.dense.weight"]
                else:
                    for val in [n, "bert." + n, "cls." + n]:
                        if val in partial:
                            update[n] = partial[val]
                        else:
                            vals = [val,val.replace('beta','bias').replace('gamma','weight')]
                            for val1 in vals:
                                if val1 in partial:
                                    update[n] = partial[val1]
                    
            self.model.bert.load_state_dict(update)
            log.info("Bert Model Weights Loaded.")
        else:
            raise ConfigError("No pre-trained BERT model is given.")

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
                "weight_decay":0.01
            },
            {
                "params": get_decay_params(self.model),
                "weight_decay": 0
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
            args_in, {}, self.task_names, "in"
        )

        for task_id in range(len(self.task_names)):
            task_features = in_by_tasks[self.task_names[task_id]][0]
            _input = {}
            if isinstance(task_features, list):
                task_features = transform(task_features)
            for elem in task_features.keys():
                _input[elem] = task_features[elem].to(self.device)

            with torch.no_grad():
                logits = self.model(
                    task_id=task_id, name=self.tasks_type[task_id],
                    **_input
                )
            if self.tasks_type[task_id] == "sequence_labeling":
                # attn_mask = _input['attention_mask']
                logits = logits.cpu()
                y_mask = _input['token_type_ids'].cpu()
                logits = token_from_subtoken(logits,y_mask)
                if self.return_probas:
                    pred = torch.nn.functional.softmax(logits, dim=-1)
                    pred = pred.detach().cpu().numpy()
                else:
                    pred = torch.argmax(logits, dim=-1)
                    seq_lengths = torch.sum(y_mask, dim=1).int()
                    pred = [p[:l] for l, p in zip(seq_lengths, pred)]
                assert isinstance(pred,list)
            elif self.tasks_type[task_id] == 'question_answering':
                logits_start, logits_end = logits[0], logits[1]
                bs, seq_len = _input['token_type_ids'].size()
                mask = torch.cat([torch.ones(bs, 1, dtype=torch.int32),
                              torch.zeros(bs, seq_len - 1, dtype=torch.int32)], dim=-1).to(self.device)
                logit_mask = _input['token_type_ids'] + mask
                logits_st = softmax_mask(logits_start, logit_mask)
                logits_end = softmax_mask(logits_end, logit_mask)

                start_probs = torch.nn.functional.softmax(logits_st, dim=-1)
                end_probs = torch.nn.functional.softmax(logits_end, dim=-1)
                # scores = torch.tensor(1) - start_probs[:, 0] * end_probs[:, 0]  # ok

                outer = torch.matmul(start_probs.view(*start_probs.size(), 1),
                                     end_probs.view(end_probs.size()[0], 1, end_probs.size()[1]))
                outer_logits = torch.exp(logits_st.view(*logits_st.size(), 1) + logits_end.view(
                    logits_end.size()[0], 1, logits_end.size()[1]))

                context_max_len = torch.max(torch.sum(_input['token_type_ids'], dim=1)).to(torch.int64)

                max_ans_length = torch.min(torch.tensor(20).to(self.device), context_max_len).to(torch.int64).item()

                outer = torch.triu(outer, diagonal=0) - torch.triu(outer, diagonal=outer.size()[1] - max_ans_length)
                outer_logits = torch.triu(outer_logits, diagonal=0) - torch.triu(
                    outer_logits, diagonal=outer_logits.size()[1] - max_ans_length)

                start_pred = torch.argmax(torch.max(outer, dim=2)[0], dim=1)
                end_pred = torch.argmax(torch.max(outer, dim=1)[0], dim=1)
                logits = torch.max(torch.max(outer_logits, dim=2)[0], dim=1)[0]
                start_pred = start_pred.detach().cpu().numpy()
                end_pred = end_pred.detach().cpu().numpy()
                pred = [(pred1, pred2) for pred1, pred2 in zip(start_pred, end_pred)]
            elif self.tasks_type[task_id] in ["classification", "regression"]:
                if self.tasks_type[task_id] == "regression":  # regression
                    pred = logits.squeeze(-1).detach().cpu().tolist()
                if self.return_probas:
                    pred = torch.nn.functional.sigmoid(logits, dim=-1)
                    pred = pred.detach().cpu().numpy()
                else:
                    logits = logits.detach().cpu().numpy()
                    pred = np.argmax(logits, axis=1)
            else:
                raise NotImplementedError(f'Unsupported type {self.tasks_type[task_id]}')
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
            args_in, {}, self.task_names, "in"
        )
        in_y_by_tasks = self._distribute_arguments_by_tasks(
            args_in_y, {}, self.task_names, "in_y"
        )

        task_features = in_by_tasks[self.task_names[task_id]][0]
        task_labels = in_y_by_tasks[self.task_names[task_id]][0]
        if isinstance(task_features, list):
            task_features = transform(task_features)
        _input = {}
        #print(task_features.keys())
        #breakpoint()
        for elem in ["input_ids", "attention_mask", "token_type_ids"]:
            if elem in task_features:
                _input[elem] = task_features[elem].to(self.device)
        
        self.optimizer.zero_grad()
        if self.tasks_type[task_id] == "regression":
            _input["labels"] = torch.tensor(
                task_labels, dtype=torch.float32).to(self.device)
        elif self.tasks_type[task_id] == "question_answering":
                # start and end logits
            _input["labels"] = [torch.tensor(label, dtype=torch.float32).to(self.device) for label in task_labels]
        else:
            _input["labels"] = torch.tensor(
                task_labels, dtype=torch.long).to(self.device)        

        if self.tasks_type[task_id] == "sequence_labeling":
            #token to subtoken
            subtoken_labels = [token_labels_to_subtoken_labels(y_el, y_mask, input_mask)
                           for y_el, y_mask, input_mask in zip(_input['labels'].detach().cpu().numpy(),
                                                               _input['token_type_ids'].detach().cpu().numpy(),
                                                               _input['attention_mask'].detach().cpu().numpy())]
            _input['labels'] = torch.from_numpy(np.array(subtoken_labels)).to(torch.int64).to(self.device)

        self.optimizer.zero_grad()
        loss, _ = self.model(
            task_id=task_id, name=self.tasks_type[task_id], **_input
        )

        loss = loss / self.gradient_accumulation_steps
        try:
            loss.backward()
        except RuntimeError:
            raise ValueError(
                f"More different classes found in task {self.task_names[task_id]} "
                f"than {self.tasks_num_classes[task_id]}")

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_norm)

        if (self.steps_taken + 1) % self.gradient_accumulation_steps == 0 or (
                self.steps_taken + 1) % self.steps_per_epoch == 0:
            self.optimizer.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()  # Update learning rate schedule
            self.optimizer.zero_grad()
        self.train_losses[task_id] = loss.item()
        self.steps_taken += 1
        return {"losses": self.train_losses}

    def _distribute_arguments_by_tasks(
            self,
            args,
            kwargs,
            task_names,
            what_to_distribute,
            in_distribution=None):
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
                f"`what_to_distribute` can be 'in' or 'in_y', {repr(what_to_distribute)} is given"
            )

        if distribution is None:
            if len(task_names) != 1:
                raise ValueError(
                    f"If no `{what_to_distribute}_distribution` is not provided there have to be only 1"
                    "task for inference")
            return {
                task_names[0]: list(kwargs.values()) if kwargs else list(args)}

        if all([isinstance(task_distr, int)
               for task_distr in distribution.values()]):
            ints = True
        elif all(
            [isinstance(task_distr, list)
             for task_distr in distribution.values()]
        ):
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

        if args:
            #print(f'Distributing {args}')
            if not ints:
                distribution = {
                    task_name: len(in_distr) for task_name,
                    in_distr in distribution.items()}
            n_distributed = sum([n_args for n_args in distribution.values()])
            if len(args) != n_distributed:
                raise ConfigError(
                    f"The number of '{what_to_distribute}' arguments of MultitaskBert does not match "
                    f"the number of distributed params according to '{what_to_distribute}_distribution' parameter. "
                    f"{len(args)} parameters are in '{what_to_distribute}' and {n_distributed} parameters are "
                    f"required '{what_to_distribute}_distribution'. "
                    f"{what_to_distribute}_distribution = {distribution}")
            values_taken = 0
            for task_name in task_names:
                n_args = distribution[task_name]
                #breakpoint()
                args_by_task[task_name] = [args[i]
                                           for i in range(values_taken, values_taken + n_args)]
                values_taken += n_args

        if kwargs:
            assert kwargs
            arg_names_used = []
            for task_name in task_names:
                in_distr = distribution[task_name]
                args_by_task[task_name] = [kwargs[arg_name]
                                           for arg_name in in_distr]
                arg_names_used += in_distr
            set_used = set(arg_names_used)
            set_all = set(kwargs.keys())
            if set_used != set_all:
                if len(set_all) > len(set_used):
                    raise ConfigError(
                        f"There are unused '{what_to_distribute}' parameters {set_all - set_used}"
                    )
                else:
                    raise ConfigError(
                        f"Some parameters exist in {set_used} but not in {set_all}"
                    )        
        return args_by_task
