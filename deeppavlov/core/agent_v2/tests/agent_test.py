from datetime import datetime
import uuid

from deeppavlov import configs, build_model
from deeppavlov.core.agent_v2.agent import Agent
from deeppavlov.core.agent_v2.state_manager import StateManager
from deeppavlov.core.agent_v2.preprocessor import Preprocessor
from deeppavlov.core.agent_v2.skill_manager import SkillManager
from deeppavlov.core.agent_v2.rest_caller import RestCaller
from deeppavlov.core.agent_v2.config import AGENT_CONFIG


ner = build_model(configs.ner.ner_rus, download=True)
faq = build_model(configs.faq.tfidf_autofaq, download=True)
sentiment = build_model(configs.classifiers.rusentiment_elmo_twitter_rnn, download=True)
utterances = ['Привет!', 'Когда началась Вторая Мировая?',
              'Привет, я бот!', '1939', 'Как дела?', 'Спасибо, бот!',
              'Хорошо, а у тебя как?', 'И у меня нормально. Когда родился Петр Первый?',
              'в 1672 году', 'спасибо', ]
print("DeepPavlov configs output:")
print(ner(utterances))
print(faq(utterances))
print(sentiment(utterances))

sm = StateManager()
preprocessor = Preprocessor(annotators={ner: ['ner.tokens', 'ner.tags'], faq: ['faq-answers', None],
                                        sentiment: 'sentiment'},
                            max_workers=4)
rest_caller = RestCaller(max_workers=AGENT_CONFIG["max_workers"])
skill_manager = SkillManager(skills_selector=None, response_selector=None, rest_caller=RestCaller)

agent = Agent(sm, preprocessor, skill_manager)

# TEST predict_annotations()
annotations = agent.predict_annotations(utterances, should_reset=[False]*len(utterances))
print("Agent output:")
print(annotations)

# TEST __call__()
u_tg_ids = ['dc96f30c-4a45-4225-8c2a-f23294f1d651', '4f5928be-27dc-4ac0-a7ac-ea76f9022636', str(uuid.uuid4())]
utts = ['Что еще скажешь интересного?', 'Бот, ты тупой', '\\start']
u_d_types = ['iphone', 'android', 'iphone']
date_times = [datetime.utcnow(), datetime.utcnow(), datetime.utcnow()]
locations = ['moscow', 'novosibirsk', 'novokuznetsk']
ch_types = ['telegram', 'telegram', 'telegram']

agent(utterances=utts, user_telegram_ids=u_tg_ids, user_device_types=u_d_types,
      date_times=date_times, locations=locations, channel_types=ch_types)
