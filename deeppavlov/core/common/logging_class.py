# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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

import json
import time
import datetime
from itertools import islice
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
from logging import getLogger

import tensorflow as tf
import wandb

from deeppavlov.core.trainers.utils import NumpyArrayEncoder
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator

log = getLogger(__name__)


class TrainLogger(ABC):
    """An abstract class for logging metrics during training process.

    There are three types of logging:
        1- StdLogger: print metrics during training
        2- TensorboardLogger: to log metrics to local file specified by log_dir in .json file.
        3- WandbLogger: Not implemented yet.

    """

    @abstractmethod
    def __init__(self):
        """
        The constructor for TrainLogger class.

        """
        raise NotImplementedError

    @abstractmethod
    def __call__(self):
        """
        Call method with metrics as parameters for logging, according to chosen method.

        """
        raise NotImplementedError

    @abstractmethod
    def print_info(self):
        """
        Print inforamtion about logging method, like the logging directory...

        """
        raise NotImplementedError


class TensorboardLogger(TrainLogger):
    """
    TensorboardLogger class for logging metrics during training process into a local folder, later using TensorBoard tool for visualizations the logged data.

    Args:
        type (str): 'train' for logging metrics of training process or 'valid' for validation process.
        log_dir (str): path to local folder to log data into.

    """

    def __init__(self, type: str, log_dir: str):
        self.type = type
        if log_dir is not None:
            self.tb_writer = tf.summary.FileWriter(log_dir)
            self.log_dir = log_dir
        else:
            self.log_dir = None

    def __call__(
        self,
        nn_trainer,
        iterator: DataLearningIterator,
        tensorboard_tag: Optional[str] = None,
        tensorboard_index: Optional[int] = None,
    ) -> dict:
        """
        override call method, for 'train' logging type, log metircs of training process to log_dir/train_log.
        for 'valid' logging type, log metrics of validation process to log_dir/valid_log.
        for 'valid' type, 'call' function saves best score on validation data, and the model parameters corresponding to the best score.

        Args:
            nn_trainer: 'NNTrainer' object which contains 'self' as variable.
            iterator: :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator` used for evaluation
            tensorboard_tag: one of two options : 'every_n_batches', 'every_n_epochs'
            tensorboard_index: one of two options: 'train_batches_seen', 'epoch' corresponding to 'tensorboard_tag' types respectively.

        Returns:
            a report dict containing calculated metrics, spent time value, and other metrics according to 'type'.

        """
        if self.type == "train":
            nn_trainer._send_event(event_name="before_log")
            if nn_trainer.log_on_k_batches == 0:
                report = {
                    "time_spent": str(
                        datetime.timedelta(
                            seconds=round(time.time() - nn_trainer.start_time + 0.5)
                        )
                    )
                }
            else:
                data = islice(
                    iterator.gen_batches(
                        nn_trainer.batch_size, data_type="train", shuffle=True
                    ),
                    nn_trainer.log_on_k_batches,
                )
                report = nn_trainer.test(
                    data, nn_trainer.train_metrics, start_time=nn_trainer.start_time
                )

            report.update(
                {
                    "epochs_done": nn_trainer.epoch,
                    "batches_seen": nn_trainer.train_batches_seen,
                    "train_examples_seen": nn_trainer.examples,
                }
            )

            metrics: List[Tuple[str, float]] = list(
                report.get("metrics", {}).items()
            ) + list(nn_trainer.last_result.items())

            report.update(nn_trainer.last_result)
            if nn_trainer.losses:
                report["loss"] = sum(nn_trainer.losses) / len(nn_trainer.losses)
                nn_trainer.losses.clear()
                metrics.append(("loss", report["loss"]))

            if (
                metrics and self.log_dir is not None
            ):  # nn_trainer.tensorboard_idx is not None
                log.info(f"logging Training metrics to {self.log_dir}")
                summary = nn_trainer._tf.Summary()

                for name, score in metrics:
                    summary.value.add(
                        tag=f"{tensorboard_tag}/{name}", simple_value=score
                    )
                self.tb_writer.add_summary(summary, tensorboard_index)
                self.tb_writer.flush()

            nn_trainer._send_event(event_name="after_train_log", data=report)
        else:
            nn_trainer._send_event(event_name="before_validation")
            report = nn_trainer.test(
                iterator.gen_batches(
                    nn_trainer.batch_size, data_type="valid", shuffle=False
                ),
                start_time=nn_trainer.start_time,
            )

            report["epochs_done"] = nn_trainer.epoch
            report["batches_seen"] = nn_trainer.train_batches_seen
            report["train_examples_seen"] = nn_trainer.examples

            metrics = list(report["metrics"].items())

            # nn_trainer.tensorboard_idx is not None:
            if tensorboard_tag is not None and self.log_dir is not None:
                log.info(f"logging Validation metrics to {self.log_dir}")
                summary = nn_trainer._tf.Summary()
                for name, score in metrics:
                    summary.value.add(
                        tag=f"{tensorboard_tag}/{name}", simple_value=score
                    )
                if tensorboard_index is None:
                    tensorboard_index = nn_trainer.train_batches_seen
                self.tb_writer.add_summary(summary, tensorboard_index)
                self.tb_writer.flush()

            m_name, score = metrics[0]

            # Update the patience
            if nn_trainer.score_best is None:
                nn_trainer.patience = 0
            else:
                if nn_trainer.improved(score, nn_trainer.score_best):
                    nn_trainer.patience = 0
                else:
                    nn_trainer.patience += 1

            # Run the validation model-saving logic
            if nn_trainer._is_initial_validation():
                log.info("Initial best {} of {}".format(m_name, score))
                nn_trainer.score_best = score
            elif nn_trainer._is_first_validation() and nn_trainer.score_best is None:
                log.info("First best {} of {}".format(m_name, score))
                nn_trainer.score_best = score
                log.info("Saving model")
                nn_trainer.save()
            elif nn_trainer.improved(score, nn_trainer.score_best):
                log.info("Improved best {} of {}".format(m_name, score))
                nn_trainer.score_best = score
                log.info("Saving model")
                nn_trainer.save()
            else:
                log.info(
                    "Did not improve on the {} of {}".format(
                        m_name, nn_trainer.score_best
                    )
                )

            report["impatience"] = nn_trainer.patience
            if nn_trainer.validation_patience > 0:
                report["patience_limit"] = nn_trainer.validation_patience

            nn_trainer._send_event(event_name="after_validation", data=report)
            nn_trainer.validation_number += 1

        return report

    def print_info(self):
        raise NotImplementedError


class StdLogger(TrainLogger):
    """
    StdLogger class for printing report about current training or validation process to stdout.

    Args:
        type: 'train' for printing report of training process or 'valid' for validation process.
        log_true (boo): if True: print of the StdLogger is provided in .json file as logging method or not. default False.

    """

    def __init__(self, type: str, log_true: bool = False):
        self.type = type
        self.log_true = log_true

    def __call__(self, report: dict) -> None:
        """
        Print report to stdout.

        Args:
            report(dict): report to log to stdout.

        """
        if self.log_true:
            report = {self.type: report}
            log.info(json.dumps(report, ensure_ascii=False, cls=NumpyArrayEncoder))

    def print_info(self):
        raise NotImplementedError


class WandbLogger(TrainLogger):
    """
    WandbLogger class for logging report about current training or validation process to WandB  ("https://wandb.ai/site").

    WandB is a central dashboard to keep track of your hyperparameters, system metrics, and predictions so you can compare models live, and share your findings.

    Args:
        key (string, optional): authentication key.

    """

    # def __init__(self, wandb_init: Optional[Dict] = None):
    def __init__(self, API_Key = None, **kwargs):
        print(kwargs)
        # wandb.login(key=wandb_init.get("API_Key", None), relogin=True)
        # wandb.login(key=kwargs.get("API_Key", None), relogin=True)
        wandb.login(key=API_Key, relogin=True)
        
        # wandb.init(
        #     anonymous="allow",
        #     project=wandb_init.get("project", None),
        #     group=wandb_init.get("group", None),
        #     job_type=wandb_init.get("job_type", None),
        #     config=wandb_init.get("config", None),
        #     name=wandb_init.get("run_name", None),
        #     id = wandb_init.get("id",None) # to resume a run
        # )
        wandb.init(**kwargs["init"])

    def __call__(self, report: dict) -> None:
        """ "
        Logging report of the training process to wandb.

        Args:
            report (dict): report to log to WandB.

        Returns:
            a report dict containing calculated metrics, spent time value, and other metrics according to 'type'.

        """
        for i in report.keys():
            if type(report[i]) == dict:
                for j in report[i].keys():
                    wandb.log({j: report[i].keys()[j]},commit = False)
            else:
                if i == "time_spent":
                    t = time.strptime(report[i], "%H:%M:%S")
                    y_seconds = int(
                        datetime.timedelta(
                            hours=t.tm_hour, minutes=t.tm_min, seconds=t.tm_sec
                        ).total_seconds()
                    )
                    wandb.log({i: y_seconds},commit = False)
                else:
                    wandb.log({i: report[i]},commit = False)
        wandb.log({},commit= True) # to log all previous logs in one step.

    def close(self):
        wandb.finish()

    def print_info(self):
        raise NotImplementedError
