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

from deeppavlov.deep import find_config
from deeppavlov.core.commands.train import train_evaluate_model_from_config
from deeppavlov.core.commands.infer import interact_model


# PIPELINE_CONFIG_PATH = 'configs/classifiers/intents_dstc2.json'
# PIPELINE_CONFIG_PATH = 'configs/classifiers/intents_snips.json'
# PIPELINE_CONFIG_PATH = 'configs/ner/ner_dstc2.json'
# PIPELINE_CONFIG_PATH = 'configs/ner/ner_rus.json'
# PIPELINE_CONFIG_PATH = 'configs/ner/slotfill_dstc2.json'
# PIPELINE_CONFIG_PATH = 'configs/error_model/brillmoore_wikitypos_en.json'
# PIPELINE_CONFIG_PATH = 'configs/error_model/brillmoore_kartaslov_ru.json'
# PIPELINE_CONFIG_PATH = 'configs/error_model/levenshtein_searcher.json'
# PIPELINE_CONFIG_PATH = 'configs/go_bot/config.json'
# PIPELINE_CONFIG_PATH = 'configs/go_bot/config_minimal.json'
# PIPELINE_CONFIG_PATH = 'configs/go_bot/config_all.json'
# PIPELINE_CONFIG_PATH = 'configs/squad/squad.json'
# PIPELINE_CONFIG_PATH = 'configs/ranking/ranking_insurance.json'
# PIPELINE_CONFIG_PATH = 'configs/seq2seq_go_bot/bot_kvret.json'
# PIPELINE_CONFIG_PATH = 'configs/odqa/en_ranker_prod.json'
# PIPELINE_CONFIG_PATH = 'configs/odqa/ru_ranker_prod.json'
# PIPELINE_CONFIG_PATH = 'configs/odqa/en_odqa_infer_prod.json'
# PIPELINE_CONFIG_PATH = 'configs/odqa/ru_odqa_infer_prod.json'
# PIPELINE_CONFIG_PATH = 'configs/odqa/ranker_test.json'
# PIPELINE_CONFIG_PATH = find_config('morpho_ru_syntagrus_train')
# PIPELINE_CONFIG_PATH = find_config('morpho_ru_syntagrus_train_pymorphy')
# PIPELINE_CONFIG_PATH = find_config('linear_models_avr')


if __name__ == '__main__':
    train_evaluate_model_from_config(PIPELINE_CONFIG_PATH)
    # interact_model(PIPELINE_CONFIG_PATH)








{
  "in": ["avr_x"],
  "fit_on": [["avr_x", "y"]],
  "out": ["y_labels"],
  "name": "support_vector_classifier",
  "main": true,
  "search": true,
  "save_path": "new_sber/checkpoints/svm.pkl",
  "load_path": "new_sber/checkpoints/svm.pkl",
  "component_name": "SVC",
  "penalty": "l2",
  "loss": "squared_hinge",
  "dual": [true, false],
  "tol": {"range": [0.00001, 0.005]},
  "C": {"range": [0.5, 2.0]},
  "fit_intercept": [true, false],
  "class_weight": [null, "balanced"],
  "max_iter": {"range": [500, 10000]}
},
{
  "in": ["avr_x"],
  "fit_on": [["avr_x", "y"]],
  "out": ["y_labels"],
  "name": "support_vector_classifier",
  "main": true,
  "search": true,
  "save_path": "new_sber/checkpoints/svm.pkl",
  "load_path": "new_sber/checkpoints/svm.pkl",
  "component_name": "SVC",
  "penalty": "l2",
  "loss": "hinge",
  "dual": true,
  "tol": {"range": [0.00001, 0.005]},
  "C": {"range": [0.5, 2.0]},
  "fit_intercept": [true, false],
  "class_weight": [null, "balanced"],
  "max_iter": {"range": [500, 10000]}
},
{
  "in": ["avr_x"],
  "fit_on": [["avr_x", "y"]],
  "out": ["y_labels"],
  "name": "random_forest",
  "main": true,
  "search": true,
  "save_path": "new_sber/checkpoints/rf.pkl",
  "load_path": "new_sber/checkpoints/rf.pkl",
  "component_name": "Random Forest",
  "n_estimators": {"range": [10, 100], "discrete": true},
  "min_samples_split": {"range": [2, 10], "discrete": true},
  "min_samples_leaf": {"range": [1, 5], "discrete": true},
  "min_weight_fraction_leaf": {"range": [0.0, 0.5]},
  "max_features": ["auto", "log2"],
  "min_impurity_split": null,
  "bootstrap": true,
  "oob_score": [false, true],
  "class_weight": [null, "balanced"]
}