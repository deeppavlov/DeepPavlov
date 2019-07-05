from typing import List
from pathlib import Path
from nltk.corpus import wordnet as wn
import pandas as pd


class RuThesaurus(object):
    """Class that finds synonyms, for russian language, based on Ruthes-lite2 thesaurus.
    Args:
        with_source_token: wheither source symbol is considered as synonyms to itself
        dir_path: path which contains ruthes_lite2 thesaurus, in csv format
    Attributes:
        with_source_token: wheither source symbol is considered as synonyms to itself
        dir_path: path to ruthes_lite2 thesaurus
        synonyms_data: pd.DataFrame object that contains synonyms relations
        text_entry_data: pd.DataFrame object that contains lemma, entry_id ans pos_tag
    """

    def __init__(self, dir_path, with_source_token: bool = False):
        self.dir_path = Path(dir_path)
        self.with_source_token = with_source_token
        self.synonyms_data = pd.read_csv(self.dir_path / "synonyms.csv", index_col=0)
        self.text_entry_data = pd.read_csv(self.dir_path / "text_entry.csv", index_col=0)
        self.to_wn_postag = {
            'NOUN': ['N', 'Num', 'NumG', 'Prtc'],
            'VERB': ['V'],
            'ADJ':  ['Adj', 'AdjGprep', 'Num', 'Pron'],
            'ADV':  ['Adv', 'Prdc', 'PrepG'],
            'NUM':  ['Num', 'NumG']
        }
        self._reset_memo()

    def _reset_memo(self):
        self.memo = dict()

    def _find_synonyms(self, lemma: str, morpho_tag) -> List[str]:
        entry_id_set = set(self.text_entry_data[self.text_entry_data['lemma'] == lemma]['entry_id'])
        concept_id_set = set()
        for entry_id in entry_id_set:
            concept_id_set.update(set(self.synonyms_data[self.synonyms_data['entry_id'] == entry_id]['concept_id']))
        syn_entry_id_set = set()
        for concept_id in concept_id_set:
            syn_entry_id_set.update(set(self.synonyms_data[self.synonyms_data['concept_id'] == concept_id]['entry_id']))
        synlist = list()
        for syn_entry_id in syn_entry_id_set:
            filter_by_pos_tag = self.text_entry_data['pos'].isin(self.to_wn_postag[morpho_tag['pos_tag']])
            filter_by_entry_id = (self.text_entry_data['entry_id'] == syn_entry_id)
            synonyms_table = self.text_entry_data[filter_by_pos_tag & filter_by_entry_id]
            synlist += list(synonyms_table['lemma'])
        return synlist

    def _filter(self, synlist: List[str], source_lemma: str) -> List[str]:
        filtered_syn = set(filter(lambda x: len(x.split('_')) == 1, synlist))
        if self.with_source_token:
            filtered_syn.add(source_lemma)
        else:
            filtered_syn.discard(source_lemma)
        return list(filtered_syn)

    def get_syn(self, lemma: str, morpho_tag) -> List[str]:
        """It returns synonyms for certain lemma
        Args:
            lemma: word for that it will search synonyms
            morpho_tags: morpho tags in UD2.0 format of filtered tokens
                         e.g. {'source_token': 'удачи',
                               'pos_tag': 'NOUN'
                               'features': {'Animacy':'Inan', 'Case':'Acc', 'Gender':'Fem', 'Number': 'Plur'}}
        Return:
             List of synonyms
        """
        if (lemma, morpho_tag['pos_tag']) in self.memo:
            return self.memo.get((lemma, morpho_tag['pos_tag']))
        lemma = lemma.upper()
        synonyms = self._find_synonyms(lemma, morpho_tag)
        synonyms = self._filter(synonyms, lemma)
        self.memo[(lemma, morpho_tag['pos_tag'])] = synonyms
        return synonyms


class EnThesaurus(object):
    """Class that finds synonyms, for english language.
    Args:
        with_source_token: wheither source symbol is considered as synonyms to itself
        without_phrase: include in result synonyms that includes several words, e.g. 'give_up'
    Attributes:
        with_source_token: wheither source symbol is considered as synonyms to itself
        without_phrase: include in result synonyms that includes several words, e.g. 'give_up'
    """

    def __init__(self, with_source_token: bool=False, without_phrase: bool=False):
        self.with_source_token = with_source_token
        self.without_phrase = without_phrase
        self.to_wn_postags = {'NOUN': wn.NOUN,
                              'VERB': wn.VERB,
                              'ADJ':  wn.ADJ,
                              'ADV':  wn.ADV}

    def _find_synonyms(self, lemma, morpho_tag):
        wn_synonyms = wn.synsets(lemma, pos=self.to_wn_postags[morpho_tag['pos_tag']])
        if morpho_tag['pos_tag'] == 'ADJ':
            wn_synonyms = wn_synonyms.extend(wn.synsets(lemma, pos='s'))
        lemmas = sum((map(lambda x: x.lemmas(), wn_synonyms)), [])
        synonyms = set(map(lambda x: x.name(), lemmas))
        return synonyms

    def _filter(self, synlist: List[str], source_lemma: str) -> List[str]:
        filtered_syn = set(filter(lambda x: len(x.split('_')) == 1, synlist)) if self.without_phrase else set(synlist)
        filtered_syn.discard(source_lemma)
        if self.with_source_token:
            filtered_syn.add(source_lemma)
        else:
            filtered_syn.discard(source_lemma)
        return list(filtered_syn)

    def get_syn(self, lemma: str, morpho_tag) -> List[str]:
        """It returns synonyms for certain word
        Args:
            lemma: word for that it will search synonyms
            morpho_tags: morpho tags in UD2.0 format of filtered tokens
                         e.g. {'source_token': 'luck', 'pos_tag': 'NOUN', 'features' {'Number': 'Sing'}}
        Return:
             List of synonyms
        """
        synonyms = self._find_synonyms(lemma, morpho_tag)
        synonyms = self._filter(synonyms, lemma)
        return synonyms