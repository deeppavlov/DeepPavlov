import pytest

from pymorphy2 import MorphAnalyzer

from deeppavlov.models.augmentation.utils.inflection import RuInflector

@pytest.fixture()
def resource_ru_inflector():
    inflector = RuInflector()
    return inflector

@pytest.fixture()
def resource_ru_inflector_force():
    inflector = RuInflector()
    return inflector

@pytest.fixture()
def resource_pymorpho():
    m = MorphAnalyzer()
    return m

@pytest.mark.parametrize("source_lemma", [('бегу', 'бег'),
                                          ('прыгнул', 'прыгнуть'),
                                          ('синих', 'синий'),
                                          ('капсулы', 'капсула'),
                                          ('быстрее', 'быстрый'),
                                          ('первее', 'первый'),
                                          ('томно', 'томно')])
def test_ru_lemma_form_without_morpho_tag(resource_ru_inflector,
                                          resource_ru_inflector_force,
                                          source_lemma):
    force_infl = resource_ru_inflector_force
    infl = resource_ru_inflector
    source, lemma = source_lemma
    assert infl.lemmatize(source) == lemma

@pytest.mark.parametrize("source_lemma_morpho", [('рыло', 'рыть', {'source_token': '',
                                                                   'pos_tag': 'VERB',
                                                                   'features': {"Aspect": 'Imp', 'Gender': 'Neut'}}),
                                                 ('рыло', 'рыло', {'source_token': '',
                                                                   'pos_tag': 'NOUN',
                                                                   'features': {"Number": "Sing"}}),
                                                 ('стекло', 'стекло', {'source_token': '',
                                                                       'pos_tag': 'NOUN',
                                                                       'features': {"Number": "Sing"}}),
                                                 ('стекло', 'стечь', {'source_token': '',
                                                                       'pos_tag': 'VERB',
                                                                       'features': {}}),
                                                 ('жарко', 'жарко', {'source_token': '',
                                                                     'pos_tag': 'ADV',
                                                                     'features': {}}),
                                                 ('жарко', 'жаркий', {'source_token': '',
                                                                     'pos_tag': 'ADJ',
                                                                     'features': {}})
                                                 ])
def test_ru_lemma_form_morpho_tag(resource_ru_inflector,
                                  resource_ru_inflector_force,
                                  source_lemma_morpho):
    force_infl = resource_ru_inflector_force
    infl = resource_ru_inflector
    source, lemma, morpho = source_lemma_morpho
    assert infl.lemmatize(source, morpho) == lemma


@pytest.mark.parametrize("morphos_score", [({'source_token': '', 'pos_tag': 'VERB', 'features': {}},
                                            {'source_token': '', 'pos_tag': 'NOUN', 'features': {}},
                                            0),
                                           ({'source_token': '', 'pos_tag': 'ADJ', 'features': {'Number': 'Plur', 'Case': 'Gen'}},
                                            {'source_token': '', 'pos_tag': 'ADJ', 'features': {'Number': 'Sing', 'Case': 'Gen'}},
                                            2)])
def test_get_similarity_morpho_tag(resource_ru_inflector, morphos_score):
    f, s, score = morphos_score
    infl = resource_ru_inflector
    assert infl.get_morpho_tag_similarity(f, s) == score

@pytest.mark.parametrize("morphos_score", [({'source_token': '', 'pos_tag': 'VERB', 'features': {}},
                                            {'source_token': '', 'pos_tag': 'NOUN', 'features': {}},
                                            0),
                                           ({'source_token': '', 'pos_tag': 'ADJ',
                                             'features': {'Number': 'Plur', 'Case': 'Gen'}},
                                            {'source_token': '', 'pos_tag': 'ADJ',
                                             'features': {'Number': 'Plur', 'Case': 'Gen'}},
                                            1),
                                           ({'source_token': '', 'pos_tag': 'ADJ',
                                             'features': {'Number': 'Plur', 'Case': 'Gen'}},
                                            {'source_token': '', 'pos_tag': 'ADJ',
                                             'features': {'Number': 'Plur', 'Case': 'Gen', 'a': 'b'}},
                                            0),
                                            ({'source_token': '', 'pos_tag': 'ADJ',
                                             'features': {'Number': 'Plur', 'Case': 'Gen', 'a': 'b'}},
                                            {'source_token': '', 'pos_tag': 'ADJ',
                                             'features': {'Number': 'Plur', 'Case': 'Gen'}},
                                            1)
                                           ])
def test_get_similarity_morpho_tag_binary(resource_ru_inflector, morphos_score):
    f, s, score = morphos_score
    infl = resource_ru_inflector
    assert infl.get_morpho_tag_binary_similarity(f, s) == score

@pytest.mark.parametrize('param', [('рыть', 'рыли', {'source_token': '',
                                                     'pos_tag': 'VERB',
                                                     'features': {'VerbForm': 'Fin',
                                                                     'Tense': 'Past',
                                                                     'Number': 'Plur',
                                                                     'Mood': 'Ind',
                                                                     'Aspect': 'Imp',
                                                                     'Voice': 'Act'}}),
                                   ('коричневый', 'коричневых', {'source_token': '',
                                                                 'pos_tag': 'ADJ',
                                                                 'features': {
                                                                     'Number': 'Plur',
                                                                     'Case': 'Gen',
                                                                     'Degree': 'Pos'
                                                                 }}),
                                   ('стать', 'стало',
                                    {
                                        'source_token': '',
                                        'pos_tag': 'VERB',
                                        'features':
                                            {
                                                'VerbForm': 'Fin',
                                                'Tense': 'Past',
                                                'Number': 'Sing',
                                                'Mood': 'Ind',
                                                'Gender': 'Neut',
                                                'Aspect': 'Perf',
                                                'Voice': 'Mid'
                                            }
                                    }),
                                   ('порассказывать', 'порассказывайте', {'source_token': '',
                                                                          'pos_tag': 'VERB',
                                                                          'features': {'VerbForm': 'Fin',
                                                                                       'Voice': 'Act',
                                                                                       'Tense': 'Fut',
                                                                                       'Person': '2',
                                                                                       'Number': 'Plur',
                                                                                       'Mood': 'Imp',
                                                                                       'Aspect': 'Imp'

                                                                          }}),
                                   ('малоизвестный', 'малоизвестные', {'source_token': '',
                                                                       'pos_tag': 'ADJ',
                                                                       'features': {'Animacy': 'Inan',
                                                                                    'Case': 'Acc',
                                                                                    'Degree': 'Pos',
                                                                                    'Number': 'Plur'

                                                                       }}),
                                   ('быстро', 'быстро', {'source_token': '',
                                                         'pos_tag': 'ADV',
                                                         'features': {'Degree': 'Pos'}}),
                                   ('желаться', 'желается', {'source_token': '',
                                                           'pos_tag': 'VERB',
                                                           'features': {'Aspect': 'Imp',
                                                                        'Mood': 'Ind',
                                                                        'Number': 'Sing',
                                                                        'Person': '3',
                                                                        'Tense': 'Pres',
                                                                        'VerbForm': 'Fin',
                                                                        'Voice': 'Mid'
                                                           }}),
                                   ('быстрый', 'быстрее', {'source_token': '',
                                                           'pos_tag': 'ADV',
                                                           'features': {'Degree': 'Cmp'}}),
                                   ('быстрый', 'быстрее', {'source_token': '',
                                                           'pos_tag': 'ADV',
                                                           'features': {'Degree': 'Cmp'}})
                                   ])
def test_ru_inflect_token(resource_ru_inflector, resource_ru_inflector_force, param):
    source, res, morpho = param
    infl = resource_ru_inflector
    force_infl = resource_ru_inflector_force
    assert force_infl.inflect_token(source, morpho) == res
