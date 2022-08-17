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

import os
from logging import getLogger
from pathlib import Path
from typing import List, Optional, Dict

import torch
from overrides import overrides
from transformers import T5ForConditionalGeneration, T5Tokenizer

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.torch_model import TorchModel


from deeppavlov.models.kbqa.fusion_in_decoder import FiDT5


logger = getLogger(__name__)


@register('torch_generative_qa_fid')
class TorchFiD(TorchModel):
    def __init__(self,
                 pretrained_transformer: str = "t5-base",
                 attention_probs_keep_prob: Optional[float] = None,
                 hidden_keep_prob: Optional[float] = None,
                 optimizer: str = "AdamW",
                 optimizer_parameters: Optional[dict] = None,
                 bert_config_file: Optional[str] = None,
                 learning_rate_drop_patience: int = 20,
                 learning_rate_drop_div: float = 2.0,
                 load_before_drop: bool = True,
                 clip_norm: Optional[float] = None,
                 min_learning_rate: float = 1e-06,
                 generate_max_length: int = 20,
                 **kwargs): 

        if not optimizer_parameters:
            optimizer_parameters = {"lr": 0.01,
                                    "weight_decay": 0.01,
                                    "betas": (0.9, 0.999),
                                    "eps": 1e-6}
        self.generate_max_length = generate_max_length

        self.attention_probs_keep_prob = attention_probs_keep_prob
        self.hidden_keep_prob = hidden_keep_prob
        self.clip_norm = clip_norm

        self.pretrained_transformer = pretrained_transformer
        self.bert_config_file = bert_config_file
        self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_transformer, return_dict=False)

        super().__init__(optimizer=optimizer,
                         optimizer_parameters=optimizer_parameters,
                         learning_rate_drop_patience=learning_rate_drop_patience,
                         learning_rate_drop_div=learning_rate_drop_div,
                         load_before_drop=load_before_drop,
                         min_learning_rate=min_learning_rate,
                         **kwargs)

    def train_on_batch(self,
                       input_ids_batch: List[List[float]],
                       attention_mask_batch: List[List[float]],
                       target_ids_batch: List[List[float]]) -> Dict:
        input_ids_batch = torch.LongTensor(input_ids_batch).to(self.device)
        attention_mask_batch = torch.LongTensor(attention_mask_batch).to(self.device)
        target_ids_batch = torch.LongTensor(target_ids_batch).to(self.device)

        input_ = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
            'labels': target_ids_batch
        }

        self.optimizer.zero_grad()

        loss = self.model(**input_)[0]
        if self.is_data_parallel:
            loss = loss.mean()
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_norm)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return {'loss': loss.item()}

    @property
    def is_data_parallel(self) -> bool:
        return isinstance(self.model, torch.nn.DataParallel)

    def __call__(self, input_ids_batch: List[List[float]], attention_mask_batch: List[List[float]]) -> List[str]:
        input_ids_batch = torch.LongTensor(input_ids_batch).to(self.device)
        attention_mask_batch = torch.LongTensor(attention_mask_batch).to(self.device)
        input_ = {
            'input_ids': input_ids_batch,
            'attention_mask': attention_mask_batch,
        }

        model = self.model.module if hasattr(self.model, "module") else self.model
        with torch.no_grad():
            answer_ids_batch = model.generate(max_length=self.generate_max_length, **input_)

        answers_batch = self.tokenizer.batch_decode(answer_ids_batch, skip_special_tokens=True)        
        return answers_batch
    
    @overrides
    def save(self, fname: Optional[str] = None):
        if fname is None:
            fname = self.save_path
        os.makedirs(fname, exist_ok=True)
        logger.info(f"Saving checkpoint to {fname}.")

        # Save model
        model_dir_path = fname
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(model_dir_path)
        
        # Save optimizer and scheduler
        optimizer_path = os.path.join(fname, "optimizer.pth.tar")
        optimizer_state = {
            "optimizer": self.optimizer.state_dict()
        }
        torch.save(optimizer_state, optimizer_path)


    def init_optimizer_from_scratch(self) -> None:
        self.optimizer = getattr(torch.optim, self.optimizer_name)(
        self.model.parameters(), **self.optimizer_parameters)
        
        if self.lr_scheduler_name is not None:
            self.lr_scheduler = getattr(torch.optim.lr_scheduler, self.lr_scheduler_name)(
                self.optimizer, **self.lr_scheduler_parameters)
        
        if self.opt.get("criterion", None):
            self.criterion = getattr(torch.nn, self.opt.get("criterion", None))()

    def init_from_scratch(self) -> None:
        logger.info(f"From pretrained {self.pretrained_transformer}.")
        self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_transformer, return_dict=False)
        t5 = T5ForConditionalGeneration.from_pretrained(self.pretrained_transformer)

        self.model = FiDT5(t5.config)
        self.model.load_t5(t5.state_dict())
        self.model.to(self.device)

        self.init_optimizer_from_scratch()
    

    def load_from_checkpoint(self, model_dir_path: str, optimizer_path: str) -> None:
        logger.info(f"Loading model from {model_dir_path}.")
        self.model = FiDT5.from_pretrained(model_dir_path)
        self.model = self.model.to(self.device)

        logger.info(f"Loading optimizer from {optimizer_path}.")
        self.init_optimizer_from_scratch()
        optimizer_state = torch.load(optimizer_path, map_location=self.device)
        self.optimizer.load_state_dict(optimizer_state["optimizer"])

    @overrides
    def load(self, fname: Optional[str] = None) -> None:
        if fname is not None:
            self.load_path = fname

        # Loading weights from checkpoint
        if self.load_path is not None:
            logger.info(f"Load path {self.load_path} is given.")
            model_dir_path = self.load_path
            optimizer_path = os.path.join(self.load_path, "optimizer.pth.tar")

            if Path(model_dir_path).exists() and Path(optimizer_path).exists():
                self.load_from_checkpoint(model_dir_path, optimizer_path)
            else:
                self.init_from_scratch()
                logger.info(f"Init from scratch. Model_path: {model_dir_path} or optimizer_path: {optimizer_path} does not exist.")
        else:
            self.init_from_scratch()
            logger.info(f"Init from scratch. Load path {self.load_path} does not exist.")

        if self.device.type == "cuda" and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)