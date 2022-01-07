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
from pathlib import Path
import datetime
from itertools import islice
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
from logging import getLogger

import tensorflow as tf
import wandb

from deeppavlov.core.trainers.utils import NumpyArrayEncoder
from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.trainers.nn_trainer import NNTrainer

log = getLogger(__name__)


class TrainLogger(ABC):
    """An abstract class for logging metrics during training process.

    There are three types of logging:
        - StdLogger: for logging report about current training and validation processes to stdout.
        - TensorboardLogger: for logging to tensorboard.
        - WandbLogger: for logging to WandB.

    """

    @abstractmethod
    def __init__() -> None:
        """
        The constructor for TrainLogger class.

        """
        raise NotImplementedError

    @abstractmethod
    def get_report(self,
                   nn_trainer: NNTrainer,
                   iterator: DataLearningIterator, type: str = None) -> dict:
        """"
        Get report about current process.
        for 'valid' type, 'get_report' function also saves best score on validation data, and the model parameters corresponding to the best score.

        Args:
            nn_trainer: 'NNTrainer' object contains parameters required for preparing the report.
            iterator: :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator` used for evaluation
            type : if "train" returns report about training process, "valid" returns report about validation process.

        Returns:
            dict contains data about current 'type' process.

        """
        if type == "train":
            if nn_trainer.log_on_k_batches == 0:
                report = {
                    "time_spent": str(
                        datetime.timedelta(
                            seconds=round(
                                time.time() - nn_trainer.start_time + 0.5)
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
                report["loss"] = sum(nn_trainer.losses) / \
                    len(nn_trainer.losses)
                nn_trainer.losses.clear()
                metrics.append(("loss", report["loss"]))

        elif type == "valid":
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

            nn_trainer.validation_number += 1
        return report

    @abstractmethod
    def __call__() -> None:
        raise NotImplementedError


class StdLogger(TrainLogger):
    """
    StdLogger class for logging report about current training and validation processes to stdout.

    Args:
        stdlogging (bool): if True, log report to stdout. 
            the object of this class with stdlogging = False can be used for validation process.

    """

    def __init__(self, stdlogging: bool = True) -> None:
        self.stdlogging = stdlogging

    def get_report(self, nn_trainer: NNTrainer, iterator: DataLearningIterator, type: str = None) -> dict:
        return super().get_report(nn_trainer=nn_trainer, iterator=iterator, type=type)

    def __call__(self, nn_trainer: NNTrainer, iterator: DataLearningIterator, type: str = None, report: Dict = None) -> dict:
        """
        override call method, to log report to stdout.

        Args:
            nn_trainer: NNTrainer object contains parameters required for preparing report. 
            iterator: :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator` used for evaluation.
            type : process type, if "train" logs report about training process, else if "valid" logs report about validation process.
            report: dictionary contains current process information, if None, use 'get_report' method to get this report.

        Returns:
            dict contains logged data to stdout.

        """
        if report is None:
            report = self.get_report(
                nn_trainer=nn_trainer, iterator=iterator, type=type)
        if self.stdlogging:
            log.info(json.dumps({type: report},
                     ensure_ascii=False, cls=NumpyArrayEncoder))
        return report


class TensorboardLogger(TrainLogger):
    """
    TensorboardLogger class for logging to tesnorboard.

    Args:
        log_dir (Path): path to local folder to log data into.

    """

    def __init__(self, log_dir: Path = None) -> None:
        self.train_log_dir = str(log_dir / 'train_log')
        self.valid_log_dir = str(log_dir / 'valid_log')
        self.tb_train_writer = tf.summary.FileWriter(self.train_log_dir)
        self.tb_valid_writer = tf.summary.FileWriter(self.valid_log_dir)

    def get_report(self, nn_trainer, iterator: DataLearningIterator, type: str = None):
        return super().get_report(nn_trainer, iterator, type=type)

    def __call__(self, nn_trainer: NNTrainer, iterator: DataLearningIterator, type: str = None, tensorboard_tag: Optional[str] = None, tensorboard_index: Optional[int] = None, report: Dict = None) -> dict:
        """
        override call method, for 'train' logging type, log metircs of training process to log_dir/train_log.
        for 'valid' logging type, log metrics of validation process to log_dir/valid_log.

        Args:
            nn_trainer: NNTrainer object contains parameters required for preparing the report.
            iterator: :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator` used for evaluation
            type : process type, if "train" logs report about training process, else if "valid" logs report about validation process.
            tensorboard_tag: one of two options : 'every_n_batches', 'every_n_epochs'
            tensorboard_index: one of two options: 'train_batches_seen', 'epoch' corresponding to 'tensorboard_tag' types respectively.
            report: dictionary contains current process information, if None, use 'get_report' method to get this report.

        Returns:
            dict contains metrics logged to tesnorboard.

        """
        if report is None:
            report = self.get_report(
                nn_trainer=nn_trainer, iterator=iterator, type=type)

        if type == "train":
            metrics: List[Tuple[str, float]] = list(
                report.get("metrics", {}).items()
            ) + list(nn_trainer.last_result.items())
            if report.get("loss", None) is not None:
                metrics.append(("loss", report["loss"]))

            if metrics and self.train_log_dir is not None:
                summary = nn_trainer._tf.Summary()

                for name, score in metrics:
                    summary.value.add(
                        tag=f"{tensorboard_tag}/{name}", simple_value=score
                    )
                self.tb_train_writer.add_summary(summary, tensorboard_index)
                self.tb_train_writer.flush()
        else:
            metrics = list(report["metrics"].items())
            if tensorboard_tag is not None and self.valid_log_dir is not None:
                summary = nn_trainer._tf.Summary()
                for name, score in metrics:
                    summary.value.add(
                        tag=f'{tensorboard_tag}/{name}', simple_value=score)
                if tensorboard_index is None:
                    tensorboard_index = nn_trainer.train_batches_seen
                self.tb_valid_writer.add_summary(summary, tensorboard_index)
                self.tb_valid_writer.flush()
        return report


class WandbLogger(TrainLogger):
    """
    WandbLogger class for logging report about current training and validation processes to WandB during training. ("https://wandb.ai/site").

    WandB is a central dashboard to keep track of your hyperparameters, system metrics, and predictions so you can compare models live, and share your findings.
    WandB doesn't support more than one run concurrently, so logging will be on "epochs" or "batches"
    If val_every_n_epochs > 0 or log_every_n_epochs > 0 in config file, logging to wandb will be on epochs.
    Otherwise if  val_every_n_batches > 0 or log_every_n_batches > 0 in config file, logging to wandb will be on batches. 
    if none of them, logging to wandb will be ignored.

    Args:
        log_on (str): if "epochs": logging to wandb on epochs, if "batches: logging on batches.
        commit_on_valid (bool): If False wandb.log just updates the current metrics dict with the row argument and metrics won't be saved until wandb.log is called with commit=True
            to commit training and validation reports with the same steps, this argument is True if logging on validation required
        **kwargs: arguments for wandb initialization, more info: https://docs.wandb.ai/ref/python/init

    """

    @staticmethod
    def login(API_Key: str = None, relogin: bool = True) -> bool:
        """"
        static method to login to wandb account, if login or init to wandb failed, logging to wandb will be ignored.

        Args:
            API_Key (str): authentication key.
            relogin (bool): if True, force relogin if already logged in.
            report(dict): dictionary contains current process information, if None, use 'get_report' method to get this report.

        Returns:
            True if login and init processes succeed, otherwise False and logging to wandb will be ignored.

        """
        try:
            return wandb.login(key=API_Key, relogin=relogin)
        except Exception as e:
            log.warning(str(e)+", logging to WandB will be ignored")
            return False

    def get_report(self, nn_trainer: NNTrainer, iterator: DataLearningIterator, type: str = None) -> dict:
        return super().get_report(nn_trainer=nn_trainer, iterator=iterator, type=type)

    def __init__(self, log_on: str = "epochs", commit_on_valid: bool = False, **kwargs) -> None:
        self.log_on = log_on  # "epochs","batches"
        self.commit_on_valid = commit_on_valid
        try:
            wandb.init(**kwargs)
            self.init_succeed = True
        except Exception as e:
            log.warning(str(e)+", logging to WandB will be ignored")
            self.init_succeed = False

    def __call__(self, nn_trainer: NNTrainer,
                 iterator: DataLearningIterator,
                 type: str = None,
                 report: Dict = None
                 ):
        """ "
        Logging report of the training process to wandb.

        Args:
            nn_trainer: 'NNTrainer' object contains parameters required for preparing the report.
            iterator: :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator` used for evaluation
            report (dict): report for logging to WandB. If None, use 'get_report' method to get this report.
            type (str) : process type, if "train" logs report about training process, else if "valid" logs report about validation process.

        Returns:
            dict contains logged data to WandB.

        """
        if report is None:
            report = self.get_report(
                nn_trainer=nn_trainer, iterator=iterator, type=type)

        logging_type = type + "/"
        for i in report.keys():
            if isinstance(report[i], dict):
                for key, value in report[i].items():
                    wandb.log(
                        {logging_type+key: value}, commit=False)
            else:
                if i == "time_spent":
                    t = time.strptime(report[i], "%H:%M:%S")
                    y_seconds = int(
                        datetime.timedelta(
                            hours=t.tm_hour, minutes=t.tm_min, seconds=t.tm_sec
                        ).total_seconds()
                    )
                    wandb.log({logging_type+i+("(s)"): y_seconds},
                              commit=False)
                else:
                    wandb.log(
                        {logging_type+i: report[i]}, commit=False)

        # if "val_every_n_epochs" is not None, we have to commit data on validation logging, otherwise on training.
        if (self.commit_on_valid and logging_type == "valid/") or (not self.commit_on_valid and logging_type == "train/"):
            wandb.log({}, commit=True)

        return report

    @staticmethod
    def close():
        """close function to commit the not commited logs and to mark a run as finished wiht wanb.finish method, and finishes uploading all data."""
        wandb.log({}, commit= True)
        wandb.finish()
