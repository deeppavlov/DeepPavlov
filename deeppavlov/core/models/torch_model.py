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

from typing import Optional
from logging import getLogger
from pathlib import Path
from copy import deepcopy
from overrides import overrides

import torch

from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.models.nn_model import NNModel

log = getLogger(__name__)


class TorchModel(NNModel):
    """Class implements torch model's main methods.

    Args:
        device: `cpu` or `cuda` device to use
        learning_rate_drop_patience: how many validations with no improvements to wait
        learning_rate_drop_div: the divider of the learning rate after `learning_rate_drop_patience` unsuccessful
            validations
        args:
        kwargs: dictionary with model parameters

    Attributes:
        device: torch device to be used
        opt: dictionary with all model parameters
        model: torch model
        epochs_done: number of epochs that were done
        optimizer: torch.optimizers instance
        criterion: torch criterion instance
        learning_rate_drop_patience: how many validations with no improvements to wait
        learning_rate_drop_div: the divider of the learning rate after `learning_rate_drop_patience` unsuccessful
            validations
    """

    def __init__(self, device: str = "gpu",
                 learning_rate_drop_patience: Optional[int] = None,
                 learning_rate_drop_div: Optional[float] = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() and device == "gpu" else "cpu")
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = None
        self.epochs_done = 0
        self.learning_rate_drop_patience = learning_rate_drop_patience
        self.learning_rate_drop_div = learning_rate_drop_div
        self.opt = deepcopy(kwargs)

        self.load()
        # we need to switch to eval mode here because by default it's in `train` mode.
        # But in case of `interact/build_model` usage, we need to have model in eval mode.
        self.model.eval()
        log.info(f"Model was successfully initialized! Model summary:\n {self.model}")

    def init_from_opt(self, model_func):
        if callable(model_func):
            self.model = model_func(**self.opt)
            self.optimizer = getattr(torch.optim, self.opt["optimizer"])(
                self.model.parameters(), **self.opt.get("optimizer_parameters", {}))
            if self.opt.get("lr_scheduler", None):
                self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.opt["lr_scheduler"])(
                    self.optimizer, **self.opt.get("lr_scheduler_parameters", {}))

            if self.opt.get("criterion", None):
                self.criterion = getattr(torch.nn, self.opt.get("criterion", None))()
        else:
            raise AttributeError("Model is not defined.")

    @overrides
    def load(self, *args, **kwargs):
        model_func = getattr(self, self.opt.get("model_name"), None)

        if self.load_path:
            log.info(f"Load path {self.load_path} is given.")
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            weights_path = Path("{}.pth.tar".format(str(self.load_path.resolve())))
            if weights_path.exists():
                log.info(f"Load path {weights_path} exists.")
                log.info(f"Initializing `{self.__class__.__name__}` from saved.")

                # firstly, initialize with random weights and previously saved parameters
                self.init_from_opt(model_func)

                # now load the weights, optimizer from saved
                log.info(f"Loading weights from {weights_path}.")
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.epochs_done = checkpoint.get("epochs_done", 0)
            else:
                log.info(f"Init from scratch. Load path {weights_path} does not exist.")
                self.init_from_opt(model_func)
        else:
            log.info(f"Init from scratch. Load path {self.load_path} is not provided.")
            self.init_from_opt(model_func)

        self.model.to(self.device)

    @overrides
    def save(self, fname: Optional[str] = None, *args, **kwargs):
        if fname is None:
            fname = self.save_path

        if not fname.parent.is_dir():
            raise ConfigError("Provided save path is incorrect!")

        weights_path = f"{fname}.pth.tar"
        log.info(f"Saving model to {weights_path}.")
        # move the model to `cpu` before saving to provide consistency
        torch.save({
            "model_state_dict": self.model.cpu().state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epochs_done": self.epochs_done
        }, Path(weights_path))
        # return it back to device (necessary if it was on `cuda`
        self.model.to(self.device)

    @overrides
    def process_event(self, event_name: str, data: dict):
        """Process event

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
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] / self.learning_rate_drop_div
