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

from typing import List, Iterable, Union, Tuple, Dict

import numpy as np

from deeppavlov.core.models.nn_model import NNModel


class SiameseModel(NNModel):
    """The class implementing base functionality for siamese neural networks.

    Args:
        batch_size: A size of a batch.
        num_context_turns: A number of ``context`` turns in data samples.
        *args: Other parameters.
        **kwargs: Other parameters.
    """

    def __init__(self,
                 batch_size: int,
                 num_context_turns: int = 1,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.batch_size = batch_size
        self.num_context_turns = num_context_turns

    def load(self, *args, **kwargs) -> None:
        pass

    def save(self, *args, **kwargs) -> None:
        pass

    def train_on_batch(self, samples_generator: Iterable[List[np.ndarray]], y: List[int]) -> float:
        """
        This method is called by trainer to make one training step on one batch.
        The number of samples returned by `samples_generator` is always equal to `batch_size`, so we need to:
        1) accumulate data for all of the inputs of the model;
        2) format inputs of a model in a proper way using `self._make_batch` function;
        3) run a model with provided inputs and ground truth labels (`y`) using `self._train_on_batch` function;
        4) return mean loss value on the batch

        Args:
            samples_generator (Iterable[List[np.ndarray]]): generator that returns list of numpy arrays
                of words of all sentences represented as integers.
                Its shape: (number_of_context_turns + 1, max_number_of_words_in_a_sentence)
            y (List[int]): tuple of labels, with shape: (batch_size, )

        Returns:
            float: value of mean loss on the batch
        """
        buf = []
        for sample in samples_generator:
            self._append_sample_to_batch_buffer(sample, buf)
        b = self._make_batch(buf)
        loss = self._train_on_batch(b, y)
        return loss

    def __call__(self, samples_generator: Iterable[List[np.ndarray]]) -> Union[np.ndarray, List[str]]:
        """
        This method is called by trainer to make one evaluation step on one batch.

        Args:
            samples_generator (Iterable[List[np.ndarray]]):  generator that returns list of numpy arrays
            of words of all sentences represented as integers.
            Has shape: (number_of_context_turns + 1, max_number_of_words_in_a_sentence)

        Returns:
            np.ndarray: predictions for the batch of samples
        """
        y_pred = []
        buf = []
        for j, sample in enumerate(samples_generator, start=1):
            n_responses = self._append_sample_to_batch_buffer(sample, buf)
            if len(buf) >= self.batch_size:
                for i in range(len(buf) // self.batch_size):
                    b = self._make_batch(buf[i * self.batch_size:(i + 1) * self.batch_size])
                    yp = self._predict_on_batch(b)
                    y_pred += list(yp)
                lenb = len(buf) % self.batch_size
                if lenb != 0:
                    buf = buf[-lenb:]
                else:
                    buf = []
        if len(buf) != 0:
            b = self._make_batch(buf)
            yp = self._predict_on_batch(b)
            y_pred += list(yp)
        y_pred = np.asarray(y_pred)
        # reshape to [batch_size, n_responses] if needed (n_responses > 1)
        y_pred = np.reshape(y_pred, (j, n_responses)) if n_responses > 1 else y_pred
        return y_pred

    def reset(self) -> None:
        pass

    def _append_sample_to_batch_buffer(self, sample: List,
                                       buf: Union[List[List[np.ndarray]], List[Tuple[np.ndarray]]]) -> int:
        context = sample[:self.num_context_turns]
        responses = sample[self.num_context_turns:]
        buf += [context + [el] for el in responses]

        return len(responses)

    def _train_on_batch(self, batch: Union[List[np.ndarray], Dict], y: List[int]) -> float:
        pass

    def _predict_on_batch(self, batch: Union[List[np.ndarray], Dict]) -> np.ndarray:
        pass

    def _predict_context_on_batch(self, batch: List[np.ndarray]) -> np.ndarray:
        pass

    def _predict_response_on_batch(self, batch: List[np.ndarray]) -> np.ndarray:
        pass

    def _make_batch(self, x: List[List[np.ndarray]]) -> List[np.ndarray]:
        b = []
        for i in range(len(x[0])):
            z = [el[i] for el in x]
            b.append(np.asarray(z))
        return b
