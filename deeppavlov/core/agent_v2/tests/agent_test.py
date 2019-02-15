from deeppavlov import configs, build_model
from deeppavlov.core.agent_v2.agent import Agent
from deeppavlov.core.agent_v2.states_manager import StatesManager
from deeppavlov.core.agent_v2.preprocessor import Preprocessor

ner = build_model(configs.ner.ner_rus, download=True)
faq = build_model(configs.faq.tfidf_autofaq, download=True)
sentiment = build_model(configs.classifiers.rusentiment_elmo_twitter_rnn, download=True)
utterances = ['Привет, мир!', 'Сбербанк подтвердил про общежитие']
print("DeepPavlov configs output:")
print(ner(utterances))
print(faq(utterances))
print(sentiment(utterances))

sm = StatesManager()
preprocessor = Preprocessor(annotators={ner: ['ner.tokens', 'ner.tags'], faq: ['faq-answers', None],
                                        sentiment: 'sentiment'},
                            max_workers=4)

agent = Agent(sm, preprocessor)
annotations = agent.predict_annotations(utterances)
print("Agent output:")
print(annotations)
