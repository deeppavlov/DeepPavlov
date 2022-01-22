import time
import datetime
from typing import Dict
from logging import getLogger

import wandb

from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.trainers.nn_trainer import NNTrainer
from deeppavlov.core.common.logging.logging_class import TrainLogger


log = getLogger(__name__)


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
        """ "
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
            log.warning(str(e) + ", logging to WandB will be ignored")
            return False

    def __init__(
        self, log_on: str = "epochs", commit_on_valid: bool = False, **kwargs
    ) -> None:
        self.log_on = log_on  # "epochs","batches"
        self.commit_on_valid = commit_on_valid
        try:
            wandb.init(**kwargs)
            self.init_succeed = True
        except Exception as e:
            log.warning(str(e) + ", logging to WandB will be ignored")
            self.init_succeed = False

    def __call__(
        self,
        nn_trainer: NNTrainer,
        iterator: DataLearningIterator,
        type: str = None,
        report: Dict = None,
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
                nn_trainer=nn_trainer, iterator=iterator, type=type
            )

        logging_type = type + "/"
        for i in report.keys():
            if isinstance(report[i], dict):
                for key, value in report[i].items():
                    wandb.log({logging_type + key: value}, commit=False)
            else:
                if i == "time_spent":
                    t = time.strptime(report[i], "%H:%M:%S")
                    y_seconds = int(
                        datetime.timedelta(
                            hours=t.tm_hour, minutes=t.tm_min, seconds=t.tm_sec
                        ).total_seconds()
                    )
                    wandb.log({logging_type + i + ("(s)"): y_seconds}, commit=False)
                else:
                    wandb.log({logging_type + i: report[i]}, commit=False)

        # if "val_every_n_epochs" is not None, we have to commit data on validation logging, otherwise on training.
        if (self.commit_on_valid and logging_type == "valid/") or (
            not self.commit_on_valid and logging_type == "train/"
        ):
            wandb.log({}, commit=True)

        return report

    @staticmethod
    def close():
        """close function to commit the not commited logs and to mark a run as finished wiht wanb.finish method, and finishes uploading all data."""
        wandb.log({}, commit=True)
        wandb.finish()
