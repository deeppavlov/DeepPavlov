from deeppavlov.core.skill.skill import Skill
from deeppavlov.core.common.file import read_json
from deeppavlov.core.commands.infer import build_model
from deeppavlov import train_model
from deeppavlov.core.common.file import find_config
from deeppavlov.skills.default_skill.default_skill import DefaultStatelessSkill

from typing import Tuple, Optional, Union
import json

class FAQSkill(Skill):
	"""Skill, matches utterances to questions, returns predefined answers.

	Allows to create skills that give answers on frequently asked questions.
	Every skill returns response and confidence.

	Args:
		data_path: Path to '.csv' file that contains two columns with Questions and Answers.
			User's utterance will be compared with Questions column and respond will be selected
			from matching row from Answers column.
		data_path: URL to '.csv' file that contains two columns with Questions and Answers.
			User's utterance will be compared with Questions column and respond will be selected
			from matching row from Answers column.
		x_col_name: Name of the column in '.csv' file, that represents Question column.
		y_col_name: Name of the column in '.csv' file, that represents Answer column.

	Attributes:
		data_path: Path to '.csv' file that contains two columns with Questions and Answers.
			User's utterance will be compared with Questions column and respond will be selected
			from matching row from Answers column.
		data_path: URL to '.csv' file that contains two columns with Questions and Answers.
			User's utterance will be compared with Questions column and respond will be selected
			from matching row from Answers column.
		x_col_name: Name of the column in '.csv' file, that represents Question column.
		y_col_name: Name of the column in '.csv' file, that represents Answer column.
	"""
	def __init__(self, data_path=None, data_url=None, x_col_name='Question', y_col_name='Answer'):
		model_config = read_json(find_config('tfidf_autofaq'))

		if data_path is None and data_url is None:
			raise ValueError("You haven't specified neither 'data_path' nor 'data_url'")
		if data_path is not None and data_url is not None:
			raise ValueError("You can't specify both 'data_path' and 'data_url'")

		if data_path != None:				
			if 'data_url' in model_config['dataset_reader']:
				del model_config['dataset_reader']['data_url']
			model_config['dataset_reader']['data_path'] = data_path

		if data_url != None:
			if 'data_path' in model_config['dataset_reader']:
				del model_config['dataset_reader']['data_path']
			model_config['dataset_reader']['data_url'] = data_url

		for i in range(len(model_config['chainer']['pipe'])):
			if 'save_path' in model_config['chainer']['pipe'][i]:
				model_config['chainer']['pipe'][i]['save_path'] = './' + model_config['chainer']['pipe'][i]['class_name'] + '.pkl'
			if 'load_path' in model_config['chainer']['pipe'][i]:
				model_config['chainer']['pipe'][i]['load_path']  = './' + model_config['chainer']['pipe'][i]['class_name'] + '.pkl'

		model = train_model(model_config)
		self.skill = DefaultStatelessSkill(model)
		

	def __call__(self, utterances_batch: list, history_batch: list,
		states_batch: Optional[list]=None) -> Tuple[list, list]:
		"""Returns skill inference result.

		Returns batches of skill inference results, estimated confidence
		levels and up to date states corresponding to incoming utterance
		batch.

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
		response, _, __ = self.skill(utterances_batch, history_batch)
		answer = ['; '.join(resp.split('; ')[:-1]) for resp in response]
		confidence = [float(resp.split('; ')[-1]) for resp in response]

		return answer, confidence





