# Copyright 2019 Neural Networks and Deep Learning lab, MIPT
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


try:
    from .accuracy import accuracy, sets_accuracy, slots_accuracy, kbqa_accuracy, \
        per_item_accuracy, per_item_dialog_accuracy, per_token_accuracy, round_accuracy
except ImportError:
    pass

try:
    from .bleu import bleu, bleu_advanced, google_bleu, per_item_bleu, per_item_dialog_bleu
except ImportError:
    pass

try:
    from .elmo_metrics import elmo_loss2ppl
except ImportError:
    pass

try:
    from .fmeasure import f1_score, ner_f1, ner_token_f1, precision_recall_f1, \
        round_f1, round_f1_macro, round_f1_weighted
except ImportError:
    pass

try:
    from .log_loss import sk_log_loss
except ImportError:
    pass

try:
    from .recall_at_k import r_at_1, r_at_2, r_at_5, r_at_10
except ImportError:
    pass

try:
    from .roc_auc_score import roc_auc_score
except ImportError:
    pass

try:
    from .squad_metrics import squad_v1_exact_match, squad_v1_f1, squad_v2_exact_match, squad_v2_f1
except ImportError:
    pass
