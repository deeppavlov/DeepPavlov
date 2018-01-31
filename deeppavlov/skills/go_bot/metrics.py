"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np


class DialogMetrics(object):
    EPS = 1e-40

    def __init__(self, n):
        self.n_actions = n
        self.reset()

    def reset(self):
        self.train_loss = 0.
        self.n_examples = 0
        self.n_dialogs = 0
        self.n_corr_dialog_actions = 0
        self.n_corr_examples = 0
        self.n_corr_dialogs = 0
        self.conf_matrix = np.zeros((self.n_actions, self.n_actions),
                                    dtype=np.float32)

    @property
    def n_corr_actions(self):
        return np.sum(np.diag(self.conf_matrix))

    @property
    def action_accuracy(self):
        return self.n_corr_actions / max(1., self.n_examples)

    @property
    def action_d_accuracy(self):
        return self.n_corr_dialog_actions / max(1., self.n_dialogs)

    @property
    def action_precisions(self):
        tp = np.diag(self.conf_matrix)
        # fp = np.sum(self.conf_matrix, axis=1) - tp
        # denom = tp + fp
        denom = np.sum(self.conf_matrix, axis=1)
        denom[np.where(denom < self.EPS)[0]] = 1
        return tp / denom

    @property
    def action_recalls(self):
        tp = np.diag(self.conf_matrix)
        # fn = np.sum(self.conf_matrix, axis=0) - tp
        # denom = tp + fn
        denom = np.sum(self.conf_matrix, axis=0)
        denom[np.where(denom < self.EPS)[0]] = 1
        return tp / denom

    def action_fs_beta(self, beta=1):
        tp = np.diag(self.conf_matrix)
        fp = np.sum(self.conf_matrix, axis=0) - tp
        fn = np.sum(self.conf_matrix, axis=1) - tp
        beta2 = beta ** 2
        beta2_1 = 1 + beta2
        denom = beta2_1 * tp + beta2 * fn + fp
        denom[np.where(denom < self.EPS)[0]] = 1.
        return beta2_1 * tp / denom

    def action_weighted_f_beta(self, beta=1):
        weights = np.sum(self.conf_matrix, axis=0)
        weights /= max(1., np.sum(weights))
        return np.sum(weights * self.action_fs_beta(beta=beta))

    @property
    def accuracy(self):
        return self.n_corr_examples / max(1., self.n_examples)

    @property
    def d_accuracy(self):
        return self.n_corr_dialogs / max(1., self.n_dialogs)

    @property
    def mean_train_loss(self):
        return self.train_loss / max(1., self.n_examples)

    def report(self):
        return ('[ dialogs:{:d} exs:{:d} mean_train_loss:{:.4f}'
                ' act_turn_acc:{:.4f}, act_dialog_acc:{:.4f}'
                ' act_weighted_f1:{:.4f}'
                ' turn_acc:{:.4f} dialog_acc:{:.4f} ]'.format(
            self.n_dialogs, self.n_examples, self.mean_train_loss,
            self.action_accuracy, self.action_d_accuracy,
            self.action_weighted_f_beta(1),
            self.accuracy, self.d_accuracy))
