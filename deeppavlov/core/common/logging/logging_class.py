# Copyright 2022 Neural Networks and Deep Learning lab, MIPT
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

import time
import datetime
from itertools import islice
from abc import ABC, abstractmethod
from typing import List, Tuple
from logging import getLogger

from deeppavlov.core.data.data_learning_iterator import DataLearningIterator
from deeppavlov.core.trainers.nn_trainer import NNTrainer


log = getLogger(__name__)


class TrainLogger(ABC):
    """An abstract class for logging metrics during training process."""

    def get_report(self, nn_trainer: NNTrainer, iterator: DataLearningIterator, type: str = None) -> dict:
        """ "
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
                report = {"time_spent": str(datetime.timedelta(
                            seconds=round(time.time() - nn_trainer.start_time + 0.5)))}
            else:
                data = islice(iterator.gen_batches(nn_trainer.batch_size, data_type="train", shuffle=True),
                    nn_trainer.log_on_k_batches,)
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

    def close():
        raise NotImplementedError
