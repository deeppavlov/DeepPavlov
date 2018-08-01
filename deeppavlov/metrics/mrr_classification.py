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
import json
from scipy.stats import rankdata

from deeppavlov.core.common.metrics_registry import register_metric
from deeppavlov.models.classifiers.utils import labels2onehot


def calc_mrr(rank):
    rank = list(map(lambda x: 1./x, rank))
    return np.mean(rank)


def mrr_from_json(fname):
    data = []
    with open(fname, encoding='utf8') as f:
        for line in f.readlines():
            data += [json.loads(line)]
    rank_i = []
    for elem in data:
        cand = elem['candidates']
        results = elem['results']
        cand_ranks = (len(results) - rankdata(results, method='average'))[cand] + 1
        rank_i.append( min(cand_ranks))
    mrr = calc_mrr(rank_i)
    return mrr


def mrr_from_dict(data):
    rank_i = []
    for elem in data:
        cand = elem['candidates']
        results = elem['results']
        cand_ranks = (len(results) - rankdata(results, method='average'))[cand] + 1
        rank_i.append( min(cand_ranks))
    mrr = calc_mrr(rank_i)
    return mrr


def make_json_predictions(fname, predictions):
    data = []
    with open(fname, encoding='utf8') as f:
        for line in f.readlines():
            data += [json.loads(line)]

    pointer = 0
    for elem_id, elem in enumerate(data):
        n = len(elem["sentences"])
        results = []
        for i in range(n):
            if elem["sentences"][i] == "":
                results.append(0)
            else:
                results.append(1 * (predictions[pointer]))
                pointer += 1
        data[elem_id]["results"] = results
    return data


@register_metric('classification_mrr')
def mrr_score(y_true, y_predicted):
    # there is hard code for selqa dataset!
    if len(y_predicted) == 66438:
        data_type = "train"
    elif len(y_predicted) == 9377:
        data_type = "dev"
    elif len(y_predicted) == 19435:
        data_type = "test"
    else:
        return 0.

    classes = np.array(list(y_predicted[0][1].keys()))
    y_true_one_hot = labels2onehot(y_true, classes)
    y_pred_probas = [y_predicted[i][1]["correct"] for i in range(len(y_predicted))]

    json_with_predictions = make_json_predictions("/home/dilyara.baymurzina/evolution_data/selqa_data/SelQA-ass-" +
                                                  data_type + ".json",
                                                  y_pred_probas)

    score = mrr_from_dict(json_with_predictions)
    return score
