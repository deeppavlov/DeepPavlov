from typing import Dict
from logging import getLogger
import json

from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.trainers.nn_trainer import NNTrainer
from deeppavlov.core.trainers.utils import NumpyArrayEncoder
from deeppavlov.core.common.logging.logging_class import TrainLogger

log = getLogger(__name__)


class StdLogger(TrainLogger):
    """
    StdLogger class for logging report about current training and validation processes to stdout.

    Args:
        stdlogging (bool): if True, log report to stdout.
            the object of this class with stdlogging = False can be used for validation process.
        **kwargs: additional parameters whose names will be logged but otherwise ignored
    """

    def __init__(self, stdlogging: bool = True, **kwargs) -> None:
        self.stdlogging = stdlogging

    def __call__(self,nn_trainer: NNTrainer, iterator: DataLearningIterator, type: str = None, report: Dict = None,
                 **kwargs) -> dict:
        """
        override call method, to log report to stdout.

        Args:
            nn_trainer: NNTrainer object contains parameters required for preparing report.
            iterator: :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator` used for evaluation.
            type : process type, if "train" logs report about training process, else if "valid" logs report about validation process.
            report: dictionary contains current process information, if None, use 'get_report' method to get this report.
            **kwargs: additional parameters whose names will be logged but otherwise ignored
        Returns:
            dict contains logged data to stdout.

        """
        if report is None:
            report = self.get_report(
                nn_trainer=nn_trainer, iterator=iterator, type=type
            )
        if self.stdlogging:
            log.info(
                json.dumps({type: report}, ensure_ascii=False, cls=NumpyArrayEncoder)
            )
        return report
        
    @staticmethod
    def close():
        log.info("Logging to Stdout completed")