import numpy as np
import torch
from torch import nn
from torchcrf import CRF as CRFbase


class CRF(CRFbase):
    """Class with Conditional Random Field from https://github.com/allenai/allennlp
       with modified training function
    """
    
    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        super().__init__(num_tags=num_tags, batch_first=batch_first)
        self.num_tags = num_tags
        nn.init.zeros_(self.transitions)
        nn.init.zeros_(self.start_transitions)
        nn.init.zeros_(self.end_transitions)
        self.stats = torch.zeros(num_tags, num_tags)
        self.zeros = torch.zeros(num_tags, num_tags)
        self.neg = torch.full((num_tags, num_tags), -1000)

    def forward(self, tags_batch: torch.LongTensor, y_masks: np.ndarray):
        seq_lengths = np.sum(y_masks, axis=1)
        for seq_len, tags_list in zip(seq_lengths, tags_batch):
            if seq_len > 1:
                for i in range(seq_len - 1):
                    self.stats[tags_list[i]][tags_list[i + 1]] += 1
        self.transitions = torch.where(self.stats > 0, self.zeros, self.neg)
