# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
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

from abc import abstractmethod
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import Optional

import torch
from overrides import overrides

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.models.nn_model import NNModel

log = getLogger(__name__)


class TorchModel(NNModel):
    """Class implements torch model's main methods.

    Args:
        device: `cpu` or `cuda` device to use
        optimizer: name of `torch.optim` optimizer
        optimizer_parameters: dictionary with optimizer parameters
        lr_scheduler: name of `torch.optim.lr_scheduler` learning rate scheduler or None
        lr_scheduler_parameters: dictionary with lr_scheduler parameters
        learning_rate_drop_patience: how many validations with no improvements to wait
        learning_rate_drop_div: the divider of the learning rate after `learning_rate_drop_patience` unsuccessful
            validations
        load_before_drop: whether to load best model before dropping learning rate or not
        min_learning_rate: min value of learning rate if learning rate decay is used
        args:
        kwargs: dictionary with other model parameters

    Attributes:
        device: `cpu` or `cuda` device to use
        opt: dictionary with all model parameters
        model: torch model
        epochs_done: number of epochs that were done
        optimizer: `torch.optim` instance
        optimizer_parameters: dictionary with optimizer parameters
        lr_scheduler: `torch.optim.lr_scheduler` instance
        lr_scheduler_parameters: dictionary with lr_scheduler parameters
        criterion: `torch.nn` criterion instance
        learning_rate_drop_patience: how many validations with no improvements to wait
        learning_rate_drop_div: the divider of the learning rate after `learning_rate_drop_patience` unsuccessful
            validations
        load_before_drop: whether to load best model before dropping learning rate or not
        min_learning_rate: min value of learning rate if learning rate decay is used
    """

    def __init__(self, device: str = "gpu",
                 optimizer: str = "AdamW",
                 optimizer_parameters: Optional[dict] = None,
                 lr_scheduler: Optional[str] = None,
                 lr_scheduler_parameters: Optional[dict] = None,
                 learning_rate_drop_patience: Optional[int] = None,
                 learning_rate_drop_div: Optional[float] = None,
                 load_before_drop: bool = True,
                 min_learning_rate: float = 0.,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = None
        self.epochs_done = 0

        if optimizer_parameters is None:
            optimizer_parameters = {"lr": 0.01}
        if lr_scheduler_parameters is None:
            lr_scheduler_parameters = dict()
        self.optimizer_name = optimizer
        self.optimizer_parameters = optimizer_parameters
        self.lr_scheduler_name = lr_scheduler
        self.lr_scheduler_parameters = lr_scheduler_parameters

        self.learning_rate_drop_patience = learning_rate_drop_patience
        self.learning_rate_drop_div = learning_rate_drop_div
        self.load_before_drop = load_before_drop
        self.min_learning_rate = min_learning_rate
        # TODO: replace opt dict with explicit arguments/structure
        self.opt = deepcopy(kwargs)

        self.load()
        # we need to switch to eval mode here because by default it's in `train` mode.
        # But in case of `interact/build_model` usage, we need to have model in eval mode.
        self.model.eval()
        log.info(f"Model was successfully initialized! Model summary:\n {self.model}")

    def init_from_opt(self, model_func: str) -> None:
        """Initialize from scratch `self.model` with the architecture built in  `model_func` method of this class
            along with `self.optimizer` as `self.optimizer_name` from `torch.optim` and parameters
            `self.optimizer_parameters`, optionally initialize `self.lr_scheduler` as `self.lr_scheduler_name` from
            `torch.optim.lr_scheduler` and parameters `self.lr_scheduler_parameters`

        Args:
            model_func: string name of this class methods

        Returns:
            None
        """
        if callable(model_func):
            self.model = model_func(**self.opt).to(self.device)
            self.optimizer = getattr(torch.optim, self.optimizer_name)(
                self.model.parameters(), **self.optimizer_parameters)
            if self.lr_scheduler_name:
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                    self.optimizer, **self.lr_scheduler_parameters)

            if self.opt.get("criterion", None):
                self.criterion = getattr(torch.nn, self.opt.get("criterion", None))()
        else:
            raise AttributeError("Model is not defined.")

    @property
    def is_data_parallel(self) -> bool:
        return isinstance(self.model, torch.nn.DataParallel)

    @overrides
    def load(self, fname: Optional[str] = None, *args, **kwargs) -> None:
        """Load model from `fname` (if `fname` is not given, use `self.load_path`) to `self.model` along with
            the optimizer `self.optimizer`, optionally `self.lr_scheduler`.
            If `fname` (if `fname` is not given, use `self.load_path`) does not exist, initialize model from scratch.

        Args:
            fname: string path to checkpoint
            *args:
            **kwargs:

        Returns:
            None
        """
        if fname is not None:
            self.load_path = fname

        model_func = getattr(self, self.opt.get("model_name", ""), None)

        if self.load_path:
            log.info(f"Load path {self.load_path} is given.")
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            weights_path = Path(self.load_path.resolve())
            weights_path = weights_path.with_suffix(f".pth.tar")
            if weights_path.exists():
                log.info(f"Load path {weights_path} exists.")
                log.info(f"Initializing `{self.__class__.__name__}` from saved.")

                # firstly, initialize with random weights and previously saved parameters
                if model_func:
                    self.init_from_opt(model_func)

                # now load the weights, optimizer from saved
                log.info(f"Loading weights from {weights_path}.")
                checkpoint = torch.load(weights_path, map_location=self.device)
                model_state = checkpoint["model_state_dict"]
                optimizer_state = checkpoint["optimizer_state_dict"]

                # load a multi-gpu model on a single device
                if not self.is_data_parallel and any(["module." in key for key in list(model_state.keys())]):
                    model_state = {key.replace("module.", ""): val for key, val in model_state.items()}

                if torch.cuda.device_count() > 1:
                    self.model.module.load_state_dict(model_state)
                else:
                    self.model.load_state_dict(model_state)
                self.optimizer.load_state_dict(optimizer_state)
                self.epochs_done = checkpoint.get("epochs_done", 0)
            elif model_func:
                log.info(f"Init from scratch. Load path {weights_path} does not exist.")
                self.init_from_opt(model_func)
        elif model_func:
            log.info(f"Init from scratch. Load path {self.load_path} is not provided.")
            self.init_from_opt(model_func)

    @overrides
    def save(self, fname: Optional[str] = None, *args, **kwargs) -> None:
        """Save torch model to `fname` (if `fname` is not given, use `self.save_path`). Checkpoint includes
            `model_state_dict`, `optimizer_state_dict`, and `epochs_done` (number of training epochs).

        Args:
            fname:
            *args:
            **kwargs:

        Returns:

        """
        if fname is None:
            fname = self.save_path

        if not fname.parent.is_dir():
            raise ConfigError("Provided save path is incorrect!")

        weights_path = Path(fname).with_suffix(f".pth.tar")
        log.info(f"Saving model to {weights_path}.")
        # move the model to `cpu` before saving to provide consistency
        if torch.cuda.device_count() > 1:
            torch.save({
                "model_state_dict": self.model.module.cpu().state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epochs_done": self.epochs_done
            }, weights_path)
        else:
            torch.save({
                "model_state_dict": self.model.cpu().state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epochs_done": self.epochs_done
            }, weights_path)
        # return it back to device (necessary if it was on `cuda`)
        self.model.to(self.device)

    @overrides
    def process_event(self, event_name: str, data: dict) -> None:
        """Process event. After epoch, increase `self.epochs_done`. After validation, decrease learning rate in
            `self.learning_rate_drop_div` times (not lower than `self.min_learning_rate`)
            if given `self.learning_rate_drop_patience`.

        Args:
            event_name: whether event is send after epoch or batch.
                    Set of values: ``"after_epoch", "after_batch"``
            data: event data (dictionary)
        Returns:
            None
        """
        if event_name == "after_epoch":
            self.epochs_done += 1

        if event_name == "after_validation" and 'impatience' in data and self.learning_rate_drop_patience:
            if data['impatience'] == self.learning_rate_drop_patience:
                log.info(f"----------Current LR is decreased in {self.learning_rate_drop_div} times----------")
                if self.load_before_drop:
                    self.load(self.save_path)
                    self.model.eval()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] / self.learning_rate_drop_div, self.min_learning_rate)

    @abstractmethod
    def train_on_batch(self, x: list, y: list):
        pass
