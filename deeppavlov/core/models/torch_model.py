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
from logging import getLogger
from pathlib import Path
from typing import Optional, Union

import torch

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.models.nn_model import NNModel

log = getLogger(__name__)


class TorchModel(NNModel):
    """Class implements torch model's main methods.

    Args:
        model: torch.nn.Model-based neural network model
        device: device to use
        optimizer: name of `torch.optim` optimizer
        optimizer_parameters: dictionary with optimizer parameters
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
        learning_rate_drop_patience: how many validations with no improvements to wait
        learning_rate_drop_div: the divider of the learning rate after `learning_rate_drop_patience` unsuccessful
            validations
        load_before_drop: whether to load best model before dropping learning rate or not
        min_learning_rate: min value of learning rate if learning rate decay is used
        clip_norm: clip gradients by norm coefficient
    """

    def __init__(self, model: torch.nn.Module,
                 device: Union[torch.device, str] = "cuda",
                 optimizer: str = "AdamW",
                 optimizer_parameters: Optional[dict] = None,
                 learning_rate_drop_patience: Optional[int] = None,
                 learning_rate_drop_div: Optional[float] = None,
                 load_before_drop: bool = True,
                 min_learning_rate: float = 1e-07,
                 clip_norm: Optional[float] = None,
                 *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.model = model
        self.device = self._init_device(device)
        self.model.to(self.device)
        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        if optimizer_parameters is None:
            optimizer_parameters = {"lr": 0.01}
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters(), **optimizer_parameters)
        self.epochs_done = 0
        self.learning_rate_drop_patience = learning_rate_drop_patience
        self.learning_rate_drop_div = learning_rate_drop_div
        self.load_before_drop = load_before_drop
        self.min_learning_rate = min_learning_rate
        self.clip_norm = clip_norm
        self.load()
        # we need to switch to eval mode here because by default it's in `train` mode.
        # But in case of `interact/build_model` usage, we need to have model in eval mode.
        self.model.eval()
        log.debug(f"Model was successfully initialized! Model summary:\n {self.model}")

    def _init_device(self, device: Union[torch.device, str]) -> torch.device:
        if device == "gpu":
            device = "cuda"
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            log.warning(f"Unable to place component {self.__class__.__name__} on GPU, "
                        "since no CUDA GPUs are available. Using CPU.")
            device = torch.device('cpu')
        return device

    @property
    def is_data_parallel(self) -> bool:
        return isinstance(self.model, torch.nn.DataParallel)

    def load(self, fname: Optional[str] = None, *args, **kwargs) -> None:
        """Load model from `fname` (if `fname` is not given, use `self.load_path`) to `self.model` along with
            the optimizer `self.optimizer`.
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

        if self.load_path:
            log.debug(f"Load path {self.load_path} is given.")
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            weights_path = Path(self.load_path.resolve())
            weights_path = weights_path.with_suffix(f".pth.tar")
            if weights_path.exists():
                log.debug(f"Load path {weights_path} exists.")
                log.debug(f"Initializing `{self.__class__.__name__}` from saved.")

                # now load the weights, optimizer from saved
                log.debug(f"Loading weights from {weights_path}.")
                checkpoint = torch.load(weights_path, map_location=self.device)
                model_state = checkpoint["model_state_dict"]
                optimizer_state = checkpoint["optimizer_state_dict"]
                # load a multi-gpu model on a single device
                if all([key.startswith("module.") for key in list(model_state.keys())]):
                    model_state = {key.replace("module.", "", 1): val for key, val in model_state.items()}

                if self.is_data_parallel:
                    self.model.module.load_state_dict(model_state)
                else:
                    self.model.load_state_dict(model_state)
                try:  # TODO: remove this try-except after hf models deep update
                    self.optimizer.load_state_dict(optimizer_state)
                except ValueError as e:
                    log.error(f'Failed to load optimizer state due to {repr(e)}')
                self.epochs_done = checkpoint.get("epochs_done", 0)
            else:
                log.warning(f"Init from scratch. Load path {weights_path} does not exist.")
        else:
            log.warning(f"Init from scratch. Load path {self.load_path} is not provided.")
        self.model.to(self.device)

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
        if self.is_data_parallel:
            model_state_dict = self.model.module.cpu().state_dict()
        else:
            model_state_dict = self.model.cpu().state_dict()
        torch.save({
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epochs_done": self.epochs_done
        }, weights_path)
        # return it back to device (necessary if it was on `cuda`)
        self.model.to(self.device)

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

    def _make_step(self, loss: torch.Tensor) -> None:
        loss.backward()
        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.optimizer.step()
