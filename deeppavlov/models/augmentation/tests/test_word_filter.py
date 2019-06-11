import pytest
from deeppavlov.models.augmentation.utils.word_filter import *

def test_isalpha_filter():
    w = WordFilter(replace_freq=1.0, isalpha_only=True, not_replaced_tokens=[])
    assert list(w.filter_isalpha_only(['ad', 'fss', '2ss', 'f%%', '\\'])) == [True, True, False, False, False]
    w = WordFilter(replace_freq=1.0, isalpha_only=False, not_replaced_tokens=[])
    assert list(w.filter_isalpha_only(['ad', 'fss', '2ss', 'f%%', '\\'])) == [True, True, True, True, True]

def test_frequence_filter():
    w = WordFilter(replace_freq=1.0, isalpha_only=True, not_replaced_tokens=[])
    assert list(w.filter_frequence([True, True, False])) == [True, True, False]
    w = WordFilter(replace_freq=0, isalpha_only=True, not_replaced_tokens=[])
    assert list(w.filter_frequence([True, True, False])) == [False, False, False]

@pytest.fixture()
def resource_ru_filter():
    w = RuWordFilter(1.0, False, ['иметь', 'яблоко'], ['NOUN', 'VERB'])
    return w

def test_ru_not_replace_filter(resource_ru_filter):
    w = resource_ru_filter
    assert list(w.filter_not_replaced_token(['имею', 'сочные', 'яблоки'], [{'pos_tag': 'VERB', 'features': {}},
                                                                           {'pos_tag': 'ADJ', 'features': {}},
                                                                           {'pos_tag': 'NOUN', 'features': {}}])) \
           == [False, True, False]

def test_ru_pos_filter(resource_ru_filter):
    w = resource_ru_filter
    assert w.filter_based_on_pos_tag([{'pos_tag': 'ADJ'},
                                      {'pos_tag': 'NOUN'},
                                      {'pos_tag': 'VERB'},
                                      {'pos_tag': 'ADV'},
                                      {'pos_tag': 'PRON'}]) == [False, True, True, False, False]

@pytest.fixture()
def resource_en_filter():
    w = EnWordFilter(1.0, False, ['apple', 'fine'], ['NOUN', 'VERB'])
    return w

def test_en_not_replace_filter(resource_en_filter):
    w = resource_en_filter
    assert list(w.filter_not_replaced_token(['apples', 'fine', 'orange'], [{'pos_tag': 'NOUN'}, {'pos_tag': 'ADJ'}, {'pos_tag': 'NOUN'}])) == [False, False, True]

def test_en_pos_filter(resource_en_filter):
    w = resource_en_filter
    assert list(w.filter_based_on_pos_tag([{'pos_tag': 'PRON', 'source_token': 'there'},
                                           {'pos_tag': 'VERB', 'source_token': 'was'},
                                           {'pos_tag': 'NOUN', 'source_token': ''},
                                           {'pos_tag': 'VERB', 'source_token': ''},
                                           {'pos_tag': 'ADJ', 'source_token': ''}])) == [False, False, True, True, False]