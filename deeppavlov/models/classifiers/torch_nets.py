from typing import List, Optional, Union

import torch
import torch.nn as nn


class ShallowAndWideCnn(nn.Module):
    def __init__(self, n_classes: int, embedding_size: int, kernel_sizes_cnn: List[int],
                 filters_cnn: Union[int, List[int]], dense_size: int, dropout_rate: float = 0.0,
                 multi_label=False, embedded_tokens=True, vocab_size: int=None, **kwargs):
        super().__init__()
        self.embedded_tokens = embedded_tokens
        self.kernel_sizes_cnn = kernel_sizes_cnn

        if not embedded_tokens and vocab_size:
            self.embedding = nn.Embedding(vocab_size, embedding_size)
        if isinstance(filters_cnn, int):
            filters_cnn = len(kernel_sizes_cnn) * [filters_cnn]

        for i in range(len(kernel_sizes_cnn)):
            setattr(self, "conv_" + str(i), nn.Conv1d(embedding_size, filters_cnn[i], kernel_sizes_cnn[i]))
            setattr(self, "bn_" + str(i), nn.BatchNorm1d(filters_cnn[i]))
            setattr(self, "relu_" + str(i), nn.ReLU())
            setattr(self, "pool_" + str(i), nn.AdaptiveMaxPool1d(1))

        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(sum(filters_cnn), dense_size)
        self.relu_dense = nn.ReLU()
        self.final_dense = nn.Linear(dense_size, n_classes)

        self.activation = nn.Sigmoid() if multi_label else nn.Softmax(dim=1)

    def forward(self, x):
        # x of shape [batch_size, number of tokens, embedding_size]
        # or x of shape [batch_size, number of tokens]
        # number of tokens is variable
        if not self.embedded_tokens:
            input = self.embedding(x)
        else:
            input = x.permute(0, 2, 1)

        # input of [batch size, embedding size, number of tokens]
        outputs = []
        for i in range(len(self.kernel_sizes_cnn)):
            output = getattr(self, "conv_" + str(i))(input)
            output = getattr(self, "bn_" + str(i))(output)
            output = getattr(self, "relu_" + str(i))(output)
            output = getattr(self, "pool_" + str(i))(output)
            output = output.squeeze(-1)
            # output of shape [batch_size, out]
            outputs.append(output)

        output = torch.cat(outputs, dim=-1)
        output = self.dropout(output)
        output = self.dense(output)
        output = self.relu_dense(output)
        output = self.dropout(output)
        output = self.final_dense(output)
        act_output = self.activation(output)
        return act_output
