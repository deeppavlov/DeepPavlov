from typing import Tuple, Optional
import json

from deeppavlov import train_model
from deeppavlov.core.skill.skill import Skill
from deeppavlov.core.common.file import read_json
from deeppavlov.core.common.file import find_config

class FAQSkill(Skill):
	"""Skill, matches utterances to questions, returns predefined answers.

	Allows to create skills that give answers on frequently asked questions.
	Skill returns response and confidence.

	Args:
		data_path: Path to '.csv' file that contains two columns with Questions and Answers.
			User's utterance will be compared with Questions column and respond will be selected
			from matching row from Answers column.
		data_url: URL to '.csv' file that contains two columns with Questions and Answers.
			User's utterance will be compared with Questions column and respond will be selected
			from matching row from Answers column.
		x_col_name: Name of the column in '.csv' file, that represents Question column.
		y_col_name: Name of the column in '.csv' file, that represents Answer column.

	Attributes:
		model: Сlassifies user's questions
	"""
	def __init__(self, data_path: str = None, data_url: str = None,
		x_col_name: str = 'Question', y_col_name: str = 'Answer') -> None:
		model_config = read_json(find_config('tfidf_autofaq'))

		if data_path is None and data_url is None:
			raise ValueError("You haven't specified neither 'data_path' nor 'data_url'")
		if data_path is not None and data_url is not None:
			raise ValueError("You can't specify both 'data_path' and 'data_url'")

		if data_path is not None:				
			if 'data_url' in model_config['dataset_reader']:
				del model_config['dataset_reader']['data_url']
			model_config['dataset_reader']['data_path'] = data_path

		if data_url is not None:
			if 'data_path' in model_config['dataset_reader']:
				del model_config['dataset_reader']['data_path']
			model_config['dataset_reader']['data_url'] = data_url

		model_config['dataset_reader']['x_col_name'] = x_col_name
		model_config['dataset_reader']['y_col_name'] = y_col_name

		for i in range(len(model_config['chainer']['pipe'])):
			if 'save_path' in model_config['chainer']['pipe'][i]:
				model_config['chainer']['pipe'][i]['save_path'] = './' + model_config['chainer']['pipe'][i]['class_name'] + '.pkl'
			if 'load_path' in model_config['chainer']['pipe'][i]:
				model_config['chainer']['pipe'][i]['load_path']  = './' + model_config['chainer']['pipe'][i]['class_name'] + '.pkl'

		self.model = train_model(model_config)		

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