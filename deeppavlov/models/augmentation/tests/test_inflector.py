import pytest
from pymorphy2 import MorphAnalyzer
from deeppavlov.models.augmentation.utils.inflector import *

@pytest.fixture()
def resource_ru_inflector():
    inflector = RuInflector(force_inflect=False)
    return inflector

@pytest.fixture()
def resource_ru_inflector_force():
    inflector = RuInflector(force_inflect=True)
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
    assert force_infl.get_lemma_form(source) == lemma
    assert infl.get_lemma_form(source) == lemma

@pytest.mark.parametrize("source_lemma_morpho", [('рыло', 'рыть', {'source_token': '',
                                                                   'pos_tag': 'VERB',
                                                                   'features': {}}),
                                                 ('рыло', 'рыло', {'source_token': '',
                                                                   'pos_tag': 'NOUN',
                                                                   'features': {}}),
                                                 ('стекло', 'стекло', {'source_token': '',
                                                                       'pos_tag': 'NOUN',
                                                                       'features': {}}),
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
    assert force_infl.get_lemma_form(source, morpho) == lemma
    assert infl.get_lemma_form(source, morpho) == lemma

def test_transform_uf20_form(resource_ru_inflector, resource_pymorpho):
    infl = resource_ru_inflector
    morph = resource_pymorpho
    parse = morph.parse('синих')[0]
    res = infl._transform_ud20_form(parse)
    assert res == {'pymorphy': parse,
                   'ud20': {'source_token': 'синих',
                            'pos_tag': 'ADJ',
                            'features': {'Number': 'Plur', 'Case': 'Gen'}}}


@pytest.mark.parametrize("morphos_score", [({'source_token': '', 'pos_tag': 'VERB', 'features': {}},
                                            {'source_token': '', 'pos_tag': 'NOUN', 'features': {}},
                                            0),
                                           ({'source_token': '', 'pos_tag': 'ADJ', 'features': {'Number': 'Plur', 'Case': 'Gen'}},
                                            {'source_token': '', 'pos_tag': 'ADJ', 'features': {'Number': 'Sing', 'Case': 'Gen'}},
                                            2)])
def test_get_similarity_morpho_tag(resource_ru_inflector, morphos_score):
    f, s, score = morphos_score
    infl = resource_ru_inflector
    assert infl._get_similarity_morpho_tag(f, s) == score

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
    assert infl._get_similarity_morpho_tag_binary(f, s) == score

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
                                   ('стать', 'стало', {'source_token': '',
                                                     'pos_tag': 'VERB',
                                                     'features': {'VerbForm': 'Fin',
                                                                  'Tense': 'Past',
                                                                  'Number': 'Sing',
                                                                  'Mood': 'Ind',
                                                                  'Gender': 'Neut',
                                                                  'Aspect': 'Perf',
                                                                  'Voice': 'Mid'}})
                                   ])
def test_inflect_token(resource_ru_inflector, resource_ru_inflector_force, param):
    source, res, morpho = param
    infl = resource_ru_inflector
    force_infl = resource_ru_inflector_force
    if infl.inflect_token(source, morpho) is not None:
        assert infl.inflect_token(source, morpho) == res
    assert force_infl.inflect_token(source, morpho) == res



