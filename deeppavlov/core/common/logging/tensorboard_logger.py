from pathlib import Path
from typing import List, Tuple, Optional, Dict
from logging import getLogger

import tensorflow as tf

from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.trainers.nn_trainer import NNTrainer
from deeppavlov.core.common.logging.logging_class import TrainLogger

log = getLogger(__name__)


class TensorboardLogger(TrainLogger):
    """
    TensorboardLogger class for logging to tesnorboard.

    Args:
        log_dir (Path): path to local folder to log data into.

    """

    def __init__(self, log_dir: Path = None) -> None:
        self.train_log_dir = str(log_dir / "train_log")
        self.valid_log_dir = str(log_dir / "valid_log")
        self.tb_train_writer = tf.summary.FileWriter(self.train_log_dir)
        self.tb_valid_writer = tf.summary.FileWriter(self.valid_log_dir)

    def __call__(
        self,
        nn_trainer: NNTrainer,
        iterator: DataLearningIterator,
        type: str = None,
        tensorboard_tag: Optional[str] = None,
        tensorboard_index: Optional[int] = None,
        report: Dict = None,
    ) -> dict:
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
                nn_trainer=nn_trainer, iterator=iterator, type=type
            )

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
                        tag=f"{tensorboard_tag}/{name}", simple_value=score
                    )
                if tensorboard_index is None:
                    tensorboard_index = nn_trainer.train_batches_seen
                self.tb_valid_writer.add_summary(summary, tensorboard_index)
                self.tb_valid_writer.flush()
        return report
