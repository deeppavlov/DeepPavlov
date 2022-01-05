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
import logging
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
    def get_report(self,
                 nn_trainer,
                 iterator: DataLearningIterator, type: str = None):
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

        else:
            # nn_trainer._send_event(event_name="before_validation")
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

            # nn_trainer._send_event(event_name="after_validation", data=report)

            nn_trainer.validation_number += 1
        return metrics, report


    @abstractmethod
    def __call__(self,
                 nn_trainer,
                 iterator: DataLearningIterator,
                 tensorboard_tag: Optional[str] = None,
                 tensorboard_index: Optional[int] = None, type: str = None):
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

class StdLogger(TrainLogger):
    """
    StdLogger class for printing report about current training and validation processes to stdout.

    Args:
        type: 'train' for printing report of training process or 'valid' for validation process.
        log_true (boo): if True: print of the StdLogger is provided in .json file as logging method or not. default False.

    """
    def __init__(self, stdlogging :bool = True):
        self.stdlogging = stdlogging
        # self.type = type

    def get_report(self, nn_trainer, iterator: DataLearningIterator, type: str = None):
        return super().get_report(nn_trainer, iterator, type=type)

    def __call__(self, nn_trainer, iterator: DataLearningIterator, type:str = None, report :Dict =None, metrics: Dict = None) -> None:
        """
        Print report to stdout.

        Args:
            report(dict): report to log to stdout.
        """
        if report is None:
            print("Calling from StdLogger:::::::::::::::::::::::::::::::::::::::::::::::")
            metrics, report = self.get_report(nn_trainer = nn_trainer, iterator = iterator, type = type)
        if self.stdlogging:
            log.info(json.dumps({type: report}, ensure_ascii=False, cls=NumpyArrayEncoder))        
        return metrics, report

    def print_info(self):
        raise NotImplementedError


class TensorboardLogger(TrainLogger):
    """
    TensorboardLogger class for logging metrics during training process into a local folder, later using TensorBoard tool for visualizations the logged data.

    Args:
        type (str): 'train' for logging metrics of training process or 'valid' for validation process.
        log_dir (str): path to local folder to log data into.

    """

    def __init__(self, log_dir: str = None):
        self.train_log_dir = str(log_dir / 'train_log')
        self.valid_log_dir = str(log_dir / 'valid_log')
        self.tb_train_writer = tf.summary.FileWriter(self.train_log_dir)
        self.tb_valid_writer = tf.summary.FileWriter(self.valid_log_dir)

    def get_report(self, nn_trainer, iterator: DataLearningIterator, type: str = None):
        return super().get_report(nn_trainer, iterator, type=type)
    
    def __call__(
        self,
        nn_trainer,
        iterator: DataLearningIterator,
        type :str = None,
        tensorboard_tag: Optional[str] = None,
        tensorboard_index: Optional[int] = None,
        report: Dict = None,
        metrics :List = None,
    ):
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
        if report is None:
            print("Calling from TensorboardLogger:::::::::::::::::::::::::::::::::::::::::::::::")
            metrics, report = self.get_report(nn_trainer = nn_trainer, iterator = iterator, type = type)
        
        # logging to tensorboard:
        if type == "train":
            if metrics and self.train_log_dir is not None:  # nn_trainer.tensorboard_idx is not None
                # log.info(f"logging Training metrics to {self.train_log_dir}")
                summary = nn_trainer._tf.Summary()

                for name, score in metrics:
                    summary.value.add(
                        tag=f"{tensorboard_tag}/{name}", simple_value=score
                    )
                # if tensorboard_index is None:
                #     tensorboard_index = nn_trainer.train_batches_seen
                self.tb_train_writer.add_summary(summary, tensorboard_index)
                self.tb_train_writer.flush()
        else:
            if tensorboard_tag is not None and self.valid_log_dir is not None:
                summary = nn_trainer._tf.Summary()
                for name, score in metrics:
                    summary.value.add(tag=f'{tensorboard_tag}/{name}', simple_value=score)
                if tensorboard_index is None:
                    tensorboard_index = nn_trainer.train_batches_seen
                self.tb_valid_writer.add_summary(summary, tensorboard_index)
                self.tb_valid_writer.flush()
        return metrics, report

    def print_info(self):
        raise NotImplementedError





class WandbLogger(TrainLogger):
    """
    WandbLogger class for logging report about current training or validation process to WandB  ("https://wandb.ai/site").

    WandB is a central dashboard to keep track of your hyperparameters, system metrics, and predictions so you can compare models live, and share your findings.

    Args:
        key (string, optional): authentication key.

    """
    
    @staticmethod
    def login(API_Key: str = None):
        return wandb.login(key=API_Key, relogin=True)

    def get_report(self, nn_trainer, iterator: DataLearningIterator, type: str = None):
        return super().get_report(nn_trainer, iterator, type=type)

    def __init__(self, log_on: str = None, commit_on_valid:bool = False, **kwargs):
        self.log_on = log_on # "epochs","batches"
        self.commit_on_valid = commit_on_valid
        wandb.init(**kwargs)

    def __call__(self,nn_trainer,
        iterator: DataLearningIterator,
        type :str = None,
        report: Dict = None,
        metrics :List = None,
        step: int = 0
        ):
        """ "
        Logging report of the training process to wandb.

        Args:
            report (dict): report to log to WandB.

        Returns:
            a report dict containing calculated metrics, spent time value, and other metrics according to 'type'.

        """
        if report is None:
            print("Calling from WandbLogger:::::::::::::::::::::::::::::::::::::::::::::::")
            metrics, report = self.get_report(nn_trainer = nn_trainer, iterator = iterator, type = type)

        logging_type = type +"/"
        for i in report.keys():
            if isinstance(report[i], dict):
            # if type(report[i]) == dict:
                for key,value in report[i].items():
                # for j in report[i].keys():
                    wandb.log(
                        {logging_type+key: value}, commit=False, step=step)
            else:
                if i == "time_spent":
                    t = time.strptime(report[i], "%H:%M:%S")
                    y_seconds = int(
                        datetime.timedelta(
                            hours=t.tm_hour, minutes=t.tm_min, seconds=t.tm_sec
                        ).total_seconds()
                    )
                    wandb.log({logging_type+i+("(s)"): y_seconds},
                              commit=False, step=step)
                else:
                    wandb.log(
                        {logging_type+i: report[i]}, commit=False, step=step)

        # if "val_every_n_epochs" is not None, we have to commit data on validation logging, otherwise on training.
        if (self.commit_on_valid and logging_type == "valid/") or (not self.commit_on_valid and logging_type == "train/"):
            # to log all previous logs in one step.
            wandb.log({}, commit=True, step=step)
        
        return metrics, report

    @staticmethod
    def close():
        wandb.log({}, commit=True)
        wandb.finish()

    def print_info(self):
        raise NotImplementedError
