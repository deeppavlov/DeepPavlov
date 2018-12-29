from typing import Tuple, Optional

from deeppavlov import train_model
from deeppavlov import build_model
from deeppavlov.core.skill.skill import Skill
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.file import find_config


class FAQSkill(Skill):
    """Skill, matches utterances to questions, returns predefined answers.

    Allows to create skills that give answers on frequently asked questions.
    Skill returns response and confidence.

    Args:
        data_path: URL or local path to '.csv' file that contains two columns with Questions and Answers.
            User's utterance will be compared with Questions column and respond will be selected
            from matching row from Answers column.
        x_col_name: Name of the column in '.csv' file, that represents Question column.
        y_col_name: Name of the column in '.csv' file, that represents Answer column.
        save_path: Path, where config file and models will be saved
        load_path: Path, where config file and models will be loaded from
        edit_dict: Dictionary of fixes in config

    Attributes:
        model: Classifies user's questions
    """

    def __init__(self, data_path: str = None, x_col_name: str = None, y_col_name: str = None, edit_dict: dict = None,
                 save_path: str = None, load_path: str = None) -> None:
        if load_path is not None and \
                (save_path is not None or data_path is not None or x_col_name is not None or y_col_name is not None):
            raise ValueError("If you specify 'load_path', you can't specify anything else, "
                             "because it leads to ambiguity")

        model_config = read_json(find_config('tfidf_autofaq'))
        if load_path is None:
            if x_col_name is not None:
                model_config['dataset_reader']['x_col_name'] = x_col_name
            if y_col_name is not None:
                model_config['dataset_reader']['y_col_name'] = y_col_name

            if save_path is None:
                save_path = './faq'
            model_config['metadata']['variables']['ROOT_PATH'] = save_path

            if data_path is not None:
                if 'data_url' in model_config['dataset_reader']:
                    del model_config['dataset_reader']['data_url']
                model_config['dataset_reader']['data_path'] = data_path

            # Need to change to recursive dict update
            model_config.update(edit_dict)
            self.model = train_model(model_config)
            print('Your model was saved at: \'' + save_path + '\'')
        else:
            model_config.update(edit_dict)
            model_config['metadata']['variables']['ROOT_PATH'] = load_path
            self.model = build_model(model_config)

    def __call__(self, utterances_batch: list, history_batch: list,
                 states_batch: Optional[list] = None) -> Tuple[list, list]:
        """Returns skill inference result.

        Returns batches of skill inference results and estimated confidence levels

        Args:
            utterances_batch: A batch of utterances of any type.
            history_batch: A batch of list typed histories for each utterance.
            states_batch: Optional. A batch of arbitrary typed states for
                each utterance.

        Returns:
            response: A batch of arbitrary typed skill inference results.
            confidence: A batch of float typed confidence levels for each of
                skill inference result.
        """
        return self.model(utterances_batch)
