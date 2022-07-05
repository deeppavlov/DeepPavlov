from logging import getLogger
from typing import List, Dict, Union, Optional
from pathlib import Path

import numpy as np
from overrides import overrides
import torch
from torch import nn
import os



from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel
from deeppavlov.models.preprocessors.torch_transformers_preprocessor import TorchTransformersMultiplechoicePreprocessor, TorchTransformersPreprocessor

log = getLogger(__name__)


prev_input = None 
class BertForMultiTask(nn.Module):
    """BERT model for classification or regression on GLUE tasks (STS-B is treated as a regression task).
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    ```
    """

    def __init__(self, tasks,task_types,
                 backbone_model='bert_base_uncased', config_file=None,
                 max_seq_len=320):

        super(BertForMultiTask, self).__init__()
        config = AutoConfig.from_pretrained(backbone_model,output_hidden_states=True,output_attentions=True)
        self.bert = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=backbone_model,
                                                                           config=config )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classes = [num_labels for num_labels in tasks]
        num_tasks = len(self.classes)
        self.max_seq_len =max_seq_len
        self.activation = nn.Tanh()
        OUT_DIM = config.hidden_size
        self.classifier = nn.ModuleList(
            [
                nn.Linear(OUT_DIM, num_labels) if task_type != 'multiple_choice'
                else nn.Linear(OUT_DIM, 1)
                for num_labels, task_type in zip(tasks, task_types)
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
            input_ids = input_ids.view(-1, input_ids.size(-1))
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        try:
            outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        except Exception as e:
            breakpoint()
            raise e
        last_hidden_state = outputs[1][-1]
        sentence_embeddings = outputs[1][0]
        first_token_tensor = last_hidden_state[:, 0]        
        pooled_output = self.bert.pooler(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        if name == 'sequence_labeling':
            final_output = self.dropout(top_hidden_state)
            logits = self.classifier[task_id](final_output)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                active_logits = logits.view(-1, self.classes[task_id])
                loss=loss_fct(active_logits, labels.view(-1))
                return loss, logits     
            else:
                return logits,outputs.hidden_states, final_output
        elif name in ['classification','regression', 'multiple_choice']:
            pooler_output = self.dropout(pooler_output)
            logits = self.classifier[task_id](pooler_output)
            if name=='multiple_choice':
                logits = logits.view((-1,self.classes[task_id]))
                if labels is not None:
                    assert len(logits)==len(labels),breakpoint()
            if labels is not None:
                if name != "regression":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits, labels)
                    if self.gsa_mode == 'no_2loss':
                        loss = loss + task_classification_loss
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
            assert span1 is not None and span2 is not None
            log.warning('SPAN CLASSIFICATION IS SUPPORTED ONLY EXPERIMENTALLY')
            input_dict = {"input":last_hidden_state, "attention_mask": attention_mask, 
                          "span1s":span1, "span2s":span2}
            if labels is not None:
                input_dict["labels"] = labels
            output_dict = self.classifier[task_id].forward(input_dict)
            if 'loss' in output_dict:
                return output_dict['loss'], output_dict['logits']
            else:
                return output_dict['logits']
        elif name == 'question_answering':
            log.warning('QUESTION ANSWERING IS SUPPORTED ONLY EXPERIMENTALLY')
            sequence_output = outputs.hidden_states[-1]        
            sequence_output = self.dropout(sequence_output)
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
        test_pipeline=True,#delete later
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
        self.include_preprocessors = include_preprocessors
        self.task_names = []
        self.test_pipeline = pipeline
        for task in tasks:
            self.task_names.append(task)
            self.tasks_num_classes.append(task['options'])
            self.tasks_type.append(task['type'])
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
        self.use_new_model=use_new_model
        self.attention_type=attention_type
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
                tasks = self.tasks_num_classes,
                task_types=self.tasks_type)

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

    def __call__(self, *args):
        """Make prediction for given features (texts).
        Args:
            features: batch of InputFeatures for all tasks
        Returns:
            predicted classes or probabilities of each class
        """
        if args.test_pipeline:
            ggg=args
            breakpoint()
            assert gg[3] == ggg[0]
            assert ggg[4] == ggg[1]
            assert ggg[5] == ggg[2]
            print('Pipelining ok')
            breakpoint()
        n_in = sum([inp for inp in self.in_distribution.values()])
        if len(args) >2:


            if type(args[0])==int or args[0] is None:
                task_id = args[0]
                features = args[1:]
                args_in, args_in_y = features[:n_in], features[n_in:]
            else:
                task_id = None
                args_in, args_in_y = args[:n_in], args[n_in:]
                
        else:
            args_in, args_in_y = args[:1], args[1:]
            task_id=0
            assert args_in_y

        self.validation_predictions = []
        in_by_tasks = self._distribute_arguments_by_tasks(args_in, {}, self.task_names, "in")
        if task_id not in [k for k in range(len(self.task_names))]:
            task_ids_to_use = [k for k in range(len(self.task_names)) if len(args_in[k])]
        else:
            task_ids_to_use = [task_id]
        for task_id in range(len(self.task_names)):
            if task_id not in task_ids_to_use:
                self.validation_predictions.append([])
            else:
                task_features = in_by_tasks[self.task_names[task_id]]
                if len(task_features) == 1 and isinstance(task_features[0],list):
                    task_features = task_features[0]
                _input = {}
                for elem in ["input_ids", "attention_mask", "token_type_ids"]:
                    if elem in task_features[0] or hasattr(task_features[0], elem):
                        try:
                            _input[elem] = [getattr(f, elem) if not isinstance(f,dict) else f[elem] for f in task_features]
                        except Exception as e:
                            breakpoint()
                            raise e
                        _input[elem] = torch.cat(_input[elem], dim=0).to(self.device)
                        if self.tasks_type[task_id] == 'multiple_choice':
                            _input[elem] = _input[elem].view((-1, _input[elem].size(-1)))
                if self.tasks_type[task_id] == "span_classification":
                    for elem in ["span1", "span2"]:
                        tmp = [getattr(f,elem) for f in task_features] 
                        try:
                            _input[elem] = torch.cat(tmp, dim=0).to(self.device)
                        except Exception as e:
                            breakpoint()
                            raise e

                with torch.no_grad():
                    logits = self.model(
                        task_id=task_id, name=self.tasks_type[task_id], **_input
                    )
                    if isinstance(logits,tuple) or logits.shape[0]==1:
                        logits=logits[0]
                if self.return_probas:
                    if not self.multilabel and self.tasks_type[task_id] != 'regression':
                        pred = torch.nn.functional.softmax(logits, dim=-1)
                    elif self.tasks_type[task_id] == 'regression':
                        pred = logits
                    pred = pred.detach().cpu().numpy()
                elif self.tasks_num_classes[task_id] > 1:
                    logits = logits.detach().cpu().numpy()
                    pred = np.argmax(logits, axis=1)
                else: 
                    # regression
                    pred = logits.squeeze(-1).detach().cpu().numpy()
                self.validation_predictions.append([pred])
        if len(args) ==1:
            return self.validation_predictions[0]
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
        if args.test_pipeline:
            ggg=args
            breakpoint()
            assert gg[3] == ggg[0]
            assert ggg[4] == ggg[1]
            assert ggg[5] == ggg[2]
            print('Pipelining ok')
            breakpoint()
        n_in = sum([inp for inp in self.in_distribution.values()])
        if len(args) >2:
            if len(args)%2!=0:
                task_id = args[0]
                features = args[1:]
                args_in, args_in_y = features[:n_in], features[n_in:]
            else:
                task_id = None
                args_in, args_in_y = args[:n_in], args[n_in:]
                
        else:
            args_in, args_in_y = args[:1], args[1:]
            task_id=0
            assert args_in_y
        in_by_tasks = self._distribute_arguments_by_tasks(
            args_in, {}, self.task_names, "in"
        )
        in_y_by_tasks = self._distribute_arguments_by_tasks(
            args_in_y, {}, self.task_names, "in_y"
        )
        if task_id == None:
            task_ids = [k for k in range(len(self.task_names)) if len(args_in[k])]
        else:
            task_ids = [task_id]
        for task_id in task_ids:
            task_features = in_by_tasks[self.task_names[task_id]]
            if len(task_features) == 1 and isinstance(task_features[0],list):
                task_features = task_features[0]
            task_labels = in_y_by_tasks[self.task_names[task_id]]

            _input = {}
            for elem in ["input_ids", "attention_mask", "token_type_ids"]:
                _input[elem] = [getattr(f, elem) if not isinstance(f,dict) else f[elem] for f in task_features]
                _input[elem] = torch.cat(_input[elem], dim=0).to(self.device)
            if self.tasks_type[task_id] == "span_classification":
                for elem in ["span1", "span2"]:
                    tmp = [getattr(f,elem) for f in task_features] 
                    try:
                        _input[elem] = torch.cat(tmp, dim=1).to(self.device)
                    except Exception as e:
                        breakpoint()
                        raise e
            if self.tasks_type[task_id] == "regression":
                _input["labels"] = torch.tensor(
                    np.array(task_labels[0], dtype=float), dtype=torch.float32
                ).to(self.device)
            else:
                _input["labels"] = torch.from_numpy(np.array(task_labels[0])).to(
                    self.device
                )
            if self.prev_id is None:
                self.prev_id = task_id
            elif self.prev_id != task_id and not self.printed:
                print('Seen samples from different tasks')
                self.printed = True
            loss, logits = self.model(
                task_id=task_id, name=self.tasks_type[task_id], **_input
            )

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

        if args and not ints:
            ints = True
            distribution = {
                task_name: len(in_distr) for task_name,
                in_distr in distribution.items()}
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
                n_args = distribution[task_name]
                args_by_task[task_name] = [values[i]
                                           for i in range(values_taken, values_taken + n_args)]
                values_taken += n_args

        else:
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
                raise ConfigError(
                    f"There are unused '{what_to_distribute}' parameters {set_all - set_used}"
                )
        return args_by_task
