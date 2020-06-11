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

from logging import getLogger
from typing import Tuple, Optional, List

from deeppavlov import build_model, train_model
from deeppavlov.configs import configs
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import read_json
from deeppavlov.core.data.utils import update_dict_recursive
from deeppavlov.deprecated.skill import Skill

log = getLogger(__name__)


class SimilarityMatchingSkill(Skill):
    """The skill matches utterances to predefined phrases and returns corresponding answers.

    The skill is based on the FAQ-alike .csv table that contains questions and corresponding responses.
    The skill returns responses and confidences.

    Args:
        data_path: URL or local path to '.csv' file that contains two columns with Utterances and Responses.
            User's utterance will be compared to the Utterances column and response will be selected
            from the Responses column.
        config_type: The selected configuration file ('tfidf_autofaq' by default).
        x_col_name: The question column name in the '.csv' file ('Question' by default).
        y_col_name: The response column name in the '.csv' file ('Answer' by default).
        save_load_path: Path, where the model will be saved or loaded from ('./similarity_matching' by default).
        edit_dict: Dictionary of edits to the selected configuration (overwrites other parameters).
        train: Should model be trained or not (True by default).

    Attributes:
        model: Classifies user's utterance
    """

    def __init__(self, data_path: Optional[str] = None, config_type: Optional[str] = 'tfidf_autofaq',
                 x_col_name: Optional[str] = 'Question', y_col_name: Optional[str] = 'Answer',
                 save_load_path: Optional[str] = './similarity_matching',
                 edit_dict: Optional[dict] = None, train: Optional[bool] = True):

        if config_type not in configs.faq:
            raise ValueError("There is no config named '{0}'. Possible options are: {1}"
                             .format(config_type, ", ".join(configs.faq.keys())))
        model_config = read_json(configs.faq[config_type])

        if x_col_name is not None:
            model_config['dataset_reader']['x_col_name'] = x_col_name
        if y_col_name is not None:
            model_config['dataset_reader']['y_col_name'] = y_col_name

        model_config['metadata']['variables']['MODELS_PATH'] = save_load_path

        if data_path is not None:
            if expand_path(data_path).exists():
                if 'data_url' in model_config['dataset_reader']:
                    del model_config['dataset_reader']['data_url']
                model_config['dataset_reader']['data_path'] = data_path
            else:
                if 'data_path' in model_config['dataset_reader']:
                    del model_config['dataset_reader']['data_path']
                model_config['dataset_reader']['data_url'] = data_path

        if edit_dict is not None:
            update_dict_recursive(model_config, edit_dict)

        if train:
            self.model = train_model(model_config, download=True)
            log.info('Your model was saved at: \'' + save_load_path + '\'')
        else:
            self.model = build_model(model_config, download=False)

    def __call__(self, utterances_batch: List[str], history_batch: List[List[str]],
                 states_batch: Optional[list] = None) -> Tuple[List[str], List[float]]:
        """It returns the skill inference result.

        Output is batches of the skill inference results and estimated confidences.

        Args:
            utterances_batch: A batch of utterances.
            history_batch: A batch of list typed histories for each utterance.
            states_batch: Optional. A batch of arbitrary typed states for
                each utterance.

        Returns:
            Batches of the skill inference results and estimated confidences.
        """
        responses, confidences = self.model(utterances_batch)

        # in case if model returns not the highest probability, but the whole distribution
        if isinstance(confidences[0], list):
            confidences = [max(c) for c in confidences]

        return responses, confidences
