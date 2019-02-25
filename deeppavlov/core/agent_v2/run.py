from datetime import datetime
import uuid

from deeppavlov import configs, build_model

from deeppavlov.core.agent_v2.agent import Agent
from deeppavlov.core.agent_v2.state_manager import StateManager
from deeppavlov.core.agent_v2.preprocessor import Preprocessor
from deeppavlov.core.agent_v2.skill_manager import SkillManager
from deeppavlov.core.agent_v2.rest_caller import RestCaller
from deeppavlov.core.agent_v2.response_selector import ConfidenceResponseSelector
from deeppavlov.core.agent_v2.config import MAX_WORKERS

import logging

from deeppavlov import build_model
from utils.telegram_utils.telegram_ui import experimental_bot

logging.getLogger('requests.packages.urllib3.connectionpool').setLevel(logging.WARNING)

ner = build_model(configs.ner.ner_rus, download=True)
faq = build_model(configs.faq.tfidf_autofaq, download=True)
sentiment = build_model(configs.classifiers.rusentiment_elmo_twitter_rnn, download=True)
state_manager = StateManager()
preprocessor = Preprocessor(annotators={ner: ['ner.tokens', 'ner.tags'], faq: ['faq-answers', None],
                                        sentiment: 'sentiment'},
                            max_workers=4)
rest_caller = RestCaller(max_workers=MAX_WORKERS)
response_selector = ConfidenceResponseSelector()
skill_manager = SkillManager(skills_selector=None, response_selector=response_selector, rest_caller=rest_caller)

agent = Agent(state_manager, preprocessor, skill_manager)


def model_function():

    def infer(messages, dialog_ids):
        u_tg_ids = [str(uuid.uuid4()) for _ in range(len(messages))]
        u_d_types = ['iphone'] * len(messages)
        date_times = [datetime.utcnow()] * len(messages)
        locations = ['moscow'] * len(messages)
        ch_types = ['telegram'] * len(messages)

        answers = agent(utterances=messages, user_telegram_ids=u_tg_ids, user_device_types=u_d_types,
                        date_times=date_times, locations=locations, channel_types=ch_types)
        return answers

    return infer


experimental_bot(model_function, token='',
                 proxy='')
