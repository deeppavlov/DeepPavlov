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

import json
import math
from logging import getLogger
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from overrides import overrides
from transformers import BertForQuestionAnswering, BertConfig, BertTokenizer
from transformers.data.processors.utils import InputFeatures

from deeppavlov import build_model
from deeppavlov.core.common.errors import ConfigError
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component
from deeppavlov.core.models.torch_model import TorchModel

logger = getLogger(__name__)


def softmax_mask(val, mask):
    INF = 1e30
    return -INF * (1 - mask.to(torch.float32)) + val


@register('torch_squad_bert_model')
class TorchBertSQuADModel(TorchModel):
    """Bert-based on PyTorch model for SQuAD-like problem setting:
    It predicts start and end position of answer for given question and context.

    [CLS] token is used as no_answer. If model selects [CLS] token as most probable
    answer, it means that there is no answer in given context.

    Start and end position of answer are predicted by linear transformation
    of Bert outputs.

    Args:
        pretrained_bert: pretrained Bert checkpoint path or key title (e.g. "bert-base-uncased")
        attention_probs_keep_prob: keep_prob for Bert self-attention layers
        hidden_keep_prob: keep_prob for Bert hidden layers
        optimizer: optimizer name from `torch.optim`
        optimizer_parameters: dictionary with optimizer's parameters,
                              e.g. {'lr': 0.1, 'weight_decay': 0.001, 'momentum': 0.9}
        bert_config_file: path to Bert configuration file, or None, if `pretrained_bert` is a string name
        learning_rate_drop_patience: how many validations with no improvements to wait
        learning_rate_drop_div: the divider of the learning rate after `learning_rate_drop_patience` unsuccessful
            validations
        load_before_drop: whether to load best model before dropping learning rate or not
        clip_norm: clip gradients by norm
        min_learning_rate: min value of learning rate if learning rate decay is used
    """

    def __init__(self,
                 pretrained_bert: str,
                 attention_probs_keep_prob: Optional[float] = None,
                 hidden_keep_prob: Optional[float] = None,
                 optimizer: str = "AdamW",
                 optimizer_parameters: dict = {"lr": 0.01, "weight_decay": 0.01,
                                               "betas": (0.9, 0.999), "eps": 1e-6},
                 bert_config_file: Optional[str] = None,
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: Optional[float] = None,
                 min_learning_rate: float = 1e-06,
                 **kwargs) -> None:

        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        self.clip_norm = clip_norm

        self.pretrained_bert = pretrained_bert
        self.bert_config_file = bert_config_file

        super().__init__(optimizer=optimizer,
                         optimizer_parameters=optimizer_parameters,
                         learning_rate_drop_patience=learning_rate_drop_patience,
                         learning_rate_drop_div=learning_rate_drop_div,
                         load_before_drop=load_before_drop,
                         min_learning_rate=min_learning_rate,
                         **kwargs)

    def train_on_batch(self, features: List[InputFeatures], y_st: List[List[int]], y_end: List[List[int]]) -> Dict:
        """Train model on given batch.
        This method calls train_op using features and labels from y_st and y_end

        Args:
            features: batch of InputFeatures instances
            y_st: batch of lists of ground truth answer start positions
            y_end: batch of lists of ground truth answer end positions

        Returns:
            dict with loss and learning_rate values

        """
        input_ids = [f.input_ids for f in features]
        input_masks = [f.attention_mask for f in features]
        input_type_ids = [f.token_type_ids for f in features]

        b_input_ids = torch.cat(input_ids, dim=0).to(self.device)
        b_input_masks = torch.cat(input_masks, dim=0).to(self.device)
        b_input_type_ids = torch.cat(input_type_ids, dim=0).to(self.device)

        y_st = [x[0] for x in y_st]
        y_end = [x[0] for x in y_end]
        b_y_st = torch.from_numpy(np.array(y_st)).to(self.device)
        b_y_end = torch.from_numpy(np.array(y_end)).to(self.device)

        self.optimizer.zero_grad()

        outputs = self.model(input_ids=b_input_ids, attention_mask=b_input_masks,
                             token_type_ids=b_input_type_ids,
                             start_positions=b_y_st, end_positions=b_y_end)
        loss = outputs[0]
        loss.backward()
        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {'loss': loss.item()}

    def __call__(self, features: List[InputFeatures]) -> Tuple[List[int], List[int], List[float], List[float]]:
        """get predictions using features as input

        Args:
            features: batch of InputFeatures instances

        Returns:
            predictions: start, end positions, start, end logits positions

        """
        input_ids = [f.input_ids for f in features]
        input_masks = [f.attention_mask for f in features]
        input_type_ids = [f.token_type_ids for f in features]

        b_input_ids = torch.cat(input_ids, dim=0).to(self.device)
        b_input_masks = torch.cat(input_masks, dim=0).to(self.device)
        b_input_type_ids = torch.cat(input_type_ids, dim=0).to(self.device)

        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = self.model(input_ids=b_input_ids, attention_mask=b_input_masks, token_type_ids=b_input_type_ids)
            logits_st, logits_end = outputs[:2]

            bs = b_input_ids.size()[0]
            seq_len = b_input_ids.size()[-1]
            mask = torch.cat([torch.ones(bs, 1, dtype=torch.int32),
                              torch.zeros(bs, seq_len - 1, dtype=torch.int32)], dim=-1).to(self.device)
            logit_mask = b_input_type_ids + mask
            logits_st = softmax_mask(logits_st, logit_mask)
            logits_end = softmax_mask(logits_end, logit_mask)

            start_probs = torch.nn.functional.softmax(logits_st, dim=-1)
            end_probs = torch.nn.functional.softmax(logits_end, dim=-1)
            scores = torch.tensor(1) - start_probs[:, 0] * end_probs[:, 0]  # ok

            outer = torch.matmul(start_probs.view(*start_probs.size(), 1),
                                 end_probs.view(end_probs.size()[0], 1, end_probs.size()[1]))
            outer_logits = torch.exp(logits_st.view(*logits_st.size(), 1) + logits_end.view(
                logits_end.size()[0], 1, logits_end.size()[1]))

            context_max_len = torch.max(torch.sum(b_input_type_ids, dim=1)).to(torch.int64)

            max_ans_length = torch.min(torch.tensor(20).to(self.device), context_max_len).to(torch.int64).item()

            outer = torch.triu(outer, diagonal=0) - torch.triu(outer, diagonal=outer.size()[1] - max_ans_length)
            outer_logits = torch.triu(outer_logits, diagonal=0) - torch.triu(
                outer_logits, diagonal=outer_logits.size()[1] - max_ans_length)

            start_pred = torch.argmax(torch.max(outer, dim=2)[0], dim=1)
            end_pred = torch.argmax(torch.max(outer, dim=1)[0], dim=1)
            logits = torch.max(torch.max(outer_logits, dim=2)[0], dim=1)[0]

        # Move logits and labels to CPU and to numpy arrays
        start_pred = start_pred.detach().cpu().numpy()
        end_pred = end_pred.detach().cpu().numpy()
        logits = logits.detach().cpu().numpy().tolist()
        scores = scores.detach().cpu().numpy().tolist()

        return start_pred, end_pred, logits, scores

    @overrides
    def load(self, fname=None):
        if fname is not None:
            self.load_path = fname

        if self.pretrained_bert and not Path(self.pretrained_bert).is_file():
            self.model = BertForQuestionAnswering.from_pretrained(
                self.pretrained_bert, output_attentions=False, output_hidden_states=False)
        elif self.bert_config_file and Path(self.bert_config_file).is_file():
            self.bert_config = BertConfig.from_json_file(str(expand_path(self.bert_config_file)))

            if self.attention_probs_keep_prob is not None:
                self.bert_config.attention_probs_dropout_prob = 1.0 - self.attention_probs_keep_prob
            if self.hidden_keep_prob is not None:
                self.bert_config.hidden_dropout_prob = 1.0 - self.hidden_keep_prob
            self.model = BertForQuestionAnswering(config=self.bert_config)
        else:
            raise ConfigError("No pre-trained BERT model is given.")

        self.model.to(self.device)
        self.optimizer = getattr(torch.optim, self.optimizer_name)(
            self.model.parameters(), **self.optimizer_parameters)
        if self.lr_scheduler_name is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)

        if self.load_path:
            logger.info(f"Load path {self.load_path} is given.")
            if isinstance(self.load_path, Path) and not self.load_path.parent.is_dir():
                raise ConfigError("Provided load path is incorrect!")

            weights_path = Path(self.load_path.resolve())
            weights_path = weights_path.with_suffix(f".pth.tar")
            if weights_path.exists():
                logger.info(f"Load path {weights_path} exists.")
                logger.info(f"Initializing `{self.__class__.__name__}` from saved.")

                # now load the weights, optimizer from saved
                logger.info(f"Loading weights from {weights_path}.")
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.epochs_done = checkpoint.get("epochs_done", 0)
            else:
                logger.info(f"Init from scratch. Load path {weights_path} does not exist.")


@register('torch_squad_bert_infer')
class TorchBertSQuADInferModel(Component):
    """This model wraps BertSQuADModel to make predictions on longer than 512 tokens sequences.

    It splits context on chunks with `max_seq_length - 3 - len(question)` length, preserving sentences boundaries.

    It reassembles batches with chunks instead of full contexts to optimize performance, e.g.,:
        batch_size = 5
        number_of_contexts == 2
        number of first context chunks == 8
        number of second context chunks == 2

        we will create two batches with 5 chunks

    For each context the best answer is selected via logits or scores from BertSQuADModel.


    Args:
        squad_model_config: path to DeepPavlov BertSQuADModel config file
        vocab_file: path to Bert vocab file
        do_lower_case: set True if lowercasing is needed
        max_seq_length: max sequence length in subtokens, including [SEP] and [CLS] tokens
        batch_size: size of batch to use during inference
        lang: either `en` or `ru`, it is used to select sentence tokenizer

    """

    def __init__(self, squad_model_config: str,
                 vocab_file: str,
                 do_lower_case: bool,
                 max_seq_length: int = 512,
                 batch_size: int = 10,
                 lang: str = 'en', **kwargs) -> None:
        config = json.load(open(squad_model_config))
        config['chainer']['pipe'][0]['max_seq_length'] = max_seq_length
        self.model = build_model(config)
        self.max_seq_length = max_seq_length

        if Path(vocab_file).is_file():
            vocab_file = str(expand_path(vocab_file))
            self.tokenizer = BertTokenizer(vocab_file=vocab_file,
                                           do_lower_case=do_lower_case)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(vocab_file, do_lower_case=do_lower_case)

        self.batch_size = batch_size

        if lang == 'en':
            from nltk import sent_tokenize
            self.sent_tokenizer = sent_tokenize
        elif lang == 'ru':
            from ru_sent_tokenize import ru_sent_tokenize
            self.sent_tokenizer = ru_sent_tokenize
        else:
            raise RuntimeError('en and ru languages are supported only')

    def __call__(self, contexts: List[str], questions: List[str], **kwargs) -> Tuple[List[str], List[int], List[float]]:
        """get predictions for given contexts and questions

        Args:
            contexts: batch of contexts
            questions: batch of questions

        Returns:
            predictions: answer, answer start position, logits or scores

        """
        batch_indices = []
        contexts_to_predict = []
        questions_to_predict = []
        predictions = {}
        for i, (context, question) in enumerate(zip(contexts, questions)):
            context_subtokens = self.tokenizer.tokenize(context)
            question_subtokens = self.tokenizer.tokenize(question)
            max_chunk_len = self.max_seq_length - len(question_subtokens) - 3
            if 0 < max_chunk_len < len(context_subtokens):
                number_of_chunks = math.ceil(len(context_subtokens) / max_chunk_len)
                sentences = self.sent_tokenizer(context)
                for chunk in np.array_split(sentences, number_of_chunks):
                    contexts_to_predict += [' '.join(chunk)]
                    questions_to_predict += [question]
                    batch_indices += [i]
            else:
                contexts_to_predict += [context]
                questions_to_predict += [question]
                batch_indices += [i]

        for j in range(0, len(contexts_to_predict), self.batch_size):
            c_batch = contexts_to_predict[j: j + self.batch_size]
            q_batch = questions_to_predict[j: j + self.batch_size]
            ind_batch = batch_indices[j: j + self.batch_size]
            a_batch, a_st_batch, logits_batch = self.model(c_batch, q_batch)
            for a, a_st, logits, ind in zip(a_batch, a_st_batch, logits_batch, ind_batch):
                if ind in predictions:
                    predictions[ind] += [(a, a_st, logits)]
                else:
                    predictions[ind] = [(a, a_st, logits)]

        answers, answer_starts, logits = [], [], []
        for ind in sorted(predictions.keys()):
            prediction = predictions[ind]
            best_answer_ind = np.argmax([p[2] for p in prediction])
            answers += [prediction[best_answer_ind][0]]
            answer_starts += [prediction[best_answer_ind][1]]
            logits += [prediction[best_answer_ind][2]]

        return answers, answer_starts, logits
