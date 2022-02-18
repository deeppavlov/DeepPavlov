import time
import datetime
from typing import Dict, Optional
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
        API_Key (str): authentication key.
        relogin (bool): if True, force relogin if already logged in.
        commit_on_valid (bool): If False wandb.log just updates the current metrics dict with the row argument and metrics won't be saved until wandb.log is called with commit=True
            to commit training and validation reports with the same steps, this argument is True if logging on validation required
        val_every_n_epochs: how often (in epochs) to validate the pipeline, ignored if negative or zero
                (default is ``-1``)
            val_every_n_batches: how often (in batches) to validate the pipeline, ignored if negative or zero
                (default is ``-1``)
            log_every_n_epochs: how often (in epochs) to calculate metrics on train data, ignored if negative or zero
                (default is ``-1``)
            log_every_n_batches: how often (in batches) to calculate metrics on train data, ignored if negative or zero
                (default is ``-1``)
        **kwargs: arguments for wandb initialization, more info: https://docs.wandb.ai/ref/python/init

    """

    @staticmethod
    def login(API_Key: str = None, relogin: bool = True) -> bool:
        """ "
        static method to login to wandb account, if login or init to wandb failed, logging to wandb will be ignored.

        Args:
            API_Key (str): authentication key.
            relogin (bool): if True, force relogin if already logged in.

        Returns:
            True if login and init processes succeed, otherwise False and logging to wandb will be ignored.

        """
        try:
            return wandb.login(key=API_Key, relogin=relogin)
        except Exception as e:
            log.warning(str(e) + ', logging to WandB will be ignored')
            return False

    def __init__(self, API_Key: str = None, relogin: bool = True, val_every_n_epochs: int = -1,
        val_every_n_batches: int = -1, log_every_n_batches: int = -1, log_every_n_epochs: int = -1, **kwargs) -> None:
        if self.login(API_Key = API_Key, relogin = relogin):
            try:
                wandb.init(**kwargs.get('init', None))
                self.init_succeed = True
                if log_every_n_epochs > 0 or val_every_n_epochs > 0:
                    self.log_on ='every_n_epochs'
                    self.commit_on_valid = val_every_n_epochs > 0
                
                elif log_every_n_batches > 0 or val_every_n_batches > 0:
                    self.log_on ='every_n_batches'
                    self.commit_on_valid = val_every_n_batches > 0
        
            except Exception as e:
                log.warning(str(e) + ', logging to WandB will be ignored')
                self.init_succeed = False
        else:
            log.warning('login to WandB failed')
            self.init_succeed = False

    def __call__(
        self,
        nn_trainer: NNTrainer,
        iterator: DataLearningIterator,
        tensorboard_tag: Optional[str] = None,
        type: str = None,
        report: Dict = None,
        **kwargs):
        """
        Logging report of the training process to wandb.

        Args:
            nn_trainer: 'NNTrainer' object contains parameters required for preparing the report.
            iterator: :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator` used for evaluation
            tensorboard_tag: one of two options : 'every_n_batches', 'every_n_epochs'
            report (dict): report for logging to WandB. If None, use 'get_report' method to get this report.
            type (str) : process type, if "train" logs report about training process, else if "valid" logs report about validation process.

        Returns:
            dict contains logged data to WandB.

        """
        if not self.init_succeed or tensorboard_tag != self.log_on:
            return None

        if report is None:
            report = self.get_report(nn_trainer=nn_trainer, iterator=iterator, type=type)
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
            not self.commit_on_valid and logging_type == "train/"):
            wandb.log({}, commit=True)

        return report

    @staticmethod
    def close():
        """close function to commit the not commited logs and to mark a run as finished wiht wanb.finish method, and finishes uploading all data."""
        wandb.log({}, commit=True)
        wandb.finish()
        log.info("Logging to W&B completed")
