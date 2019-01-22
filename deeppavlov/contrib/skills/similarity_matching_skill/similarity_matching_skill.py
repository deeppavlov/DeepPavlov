from typing import Tuple, Optional, List
import os

from deeppavlov import train_model
from deeppavlov import build_model
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.skill.skill import Skill
from deeppavlov.core.common.file import read_json
from deeppavlov.configs import configs
from deeppavlov.core.data.utils import update_dict_recursive

log = get_logger(__name__)


class SimilarityMatchingSkill(Skill):
    """Skill, matches utterances to predefined phrases, returns corresponding response.

    Allows to create skills that gives response to corresponding user's utterance
    Skill returns response and confidence.

    Args:
        data_path: URL or local path to '.csv' file that contains two columns with Utterances and Responses.
            User's utterance will be compared with Utterances column and response will be selected
            from matching row from Responses column.
        x_col_name: Name of the column in '.csv' file, that represents Utterances column.
        y_col_name: Name of the column in '.csv' file, that represents Responses column.
        save_load_path: Path, where model will be saved or loaded from.
        edit_dict: Dictionary of edits in config (has higher prior, than previous arguments).
        train: Should model be trained or not.

    Attributes:
        model: Classifies user's utterance
    """

    def __init__(self, data_path: Optional[str] = None,
                 x_col_name: Optional[str] = None, y_col_name: Optional[str] = None,
                 save_load_path: Optional[str] = None, edit_dict: Optional[dict] = None, train: bool = True):

        model_config = read_json(configs.faq.tfidf_autofaq)
        if x_col_name is not None:
            model_config['dataset_reader']['x_col_name'] = x_col_name
        if y_col_name is not None:
            model_config['dataset_reader']['y_col_name'] = y_col_name

        if save_load_path is None:
            save_load_path = './intent_matching'
        model_config['metadata']['variables']['ROOT_PATH'] = save_load_path

        if data_path is not None:
            if os.path.exists(data_path):
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
            self.model = train_model(model_config)
            log.info('Your model was saved at: \'' + save_load_path + '\'')
        else:
            self.model = build_model(model_config)

    def __call__(self, utterances_batch: List[str], history_batch: List[List[str]],
                 states_batch: Optional[list] = None) -> Tuple[List[str], List[float]]:
        """Returns skill inference result.

        Returns batches of skill inference results and estimated confidences

        Args:
            utterances_batch: A batch of utterances of any type.
            history_batch: A batch of list typed histories for each utterance.
            states_batch: Optional. A batch of arbitrary typed states for
                each utterance.

        Returns:
            Batches of skill inference results and estimated confidences
        """
        return self.model(utterances_batch)
