import pytest
from deeppavlov.models.augmentation.utils.thesaurus_wrapper import *

@pytest.fixture()
def resourse_ru_thes_with_syns():
    thes = RuThesaurus('/home/azat/.deeppavlov/downloads/ruthes_lite2', True)
    return thes

@pytest.fixture()
def resource_ru_thes():
    thes = RuThesaurus('/home/azat/.deeppavlov/downloads/ruthes_lite2', False)
    return thes

def test_ru_not_found():
    with pytest.raises(FileNotFoundError):
        RuThesaurus(dir_path='./', with_source_token=True)

def test_ru_find_syn(resourse_ru_thes_with_syns):
    thes = resourse_ru_thes_with_syns
    assert set(thes._find_synonyms('РАБ', {'pos_tag': 'NOUN'})) == set(['НЕВОЛЬНИК', 'НЕВОЛЬНИЦА', 'РАБ', 'РАБЫНЯ'])

def test_ru_getsyn(resource_ru_thes):
    thes = resource_ru_thes
    assert set(thes.get_syn('РАБ', {'pos_tag': 'NOUN'})) == set(['НЕВОЛЬНИК', 'НЕВОЛЬНИЦА', 'РАБЫНЯ'])

def test_ru_filter(resourse_ru_thes_with_syns):
    thes = resourse_ru_thes_with_syns
    assert set(thes._filter(['выа аы', 'аав', 'ННН'], 'ННН')) == set(['выа аы', 'аав', 'ННН'])

def test_ru_filter(resource_ru_thes):
    thes = resource_ru_thes
    assert set(thes._filter(['выа аы', 'аав', 'ННН'], 'ННН')) == set(['выа аы', 'аав'])

@pytest.fixture()
def resource_en_thes_with_syn():
    thes = EnThesaurus(with_source_token=True)
    return thes

@pytest.fixture()
def resource_en_thes():
    thes = EnThesaurus(with_source_token=False)
    return thes

def test_en_find_syn(resource_en_thes_with_syn):
    thes = resource_en_thes_with_syn
    assert set(thes._find_synonyms('frog', {'pos_tag': 'NOUN'})) \
           == set(['frog', 'toad', 'toad_frog', 'anuran', 'batrachian', 'salientian', 'Gaul'])

def test_en_filter(resource_en_thes_with_syn):
    thes = resource_en_thes_with_syn
    assert set(thes._filter(['frog', 'toad', 'toad_frog', 'anuran', 'batrachian', 'salientian', 'Gaul'], 'frog')) \
           == set(['frog', 'toad_frog', 'toad', 'anuran', 'batrachian', 'salientian', 'Gaul'])