import pymorphy2
from typing import List
#import pattern.en as en
import xml.etree.ElementTree as ET
from pathlib import Path
import time


class CaseSaver():
    """
    
    """

    def __init__(self, cases: dict = None):
        if cases == None:
            self.cases = {
                "lower": lambda x: x.lower(),
                "capitalize": lambda x: x.capitalize(),
                "upper": lambda x: x.upper()
            }
            self.default_case = lambda x: x.lower()
        else:
            self.cases = cases


    def _determine_case(self, token):
        for case in self.cases:
            if token == self.cases[case](token):
                return case
        return None
    

    def save_cases(self, tokens: List[str]):
        self.saved_cases = [self._determine_case(token) for token in tokens]


    def _rest_case(self, token: str, case: str):
        if case == None:
            return self.default_case(token)
        return self.cases[case](token)


    def rest_cases(self, tokens: List[str]) -> List[str]:
        return [self._rest_case(token, saved_case) for saved_case, token in zip(self.saved_cases, tokens)]


class RuInflector():
    """
    """

    def __init__():
        self.morph = pymorphy2.MorphAnalyzer()

    
    def get_morpho_tag(self, token: str) -> pymorphy2.analyzer.Parse:
        return self.morph.parse(token)[0]


    def get_lemma_form(self, token: str) -> str:
        return self.morph.parse(token)[0].normal_form


    def inflect_token(self, token: str, morpho_tag: pymorphy2.analyzer.Parse):
        return self.morph.parse(token)[0].inflect(morpho_tag)


    def force_inflect_token(self, token: str, morpho_tag: pymorphy2.analyzer.Parse):
        form = _extract_morpho_requirements(morpho_tag)
        return self.morph.parse(token)[0].inflect(form)


    def _extract_morpho_requirements(self, tag: pymorphy2.analyzer.Parse):
        #Not all morphems are need for inflection
        if tag.POS in ["NOUN"]:
            keys = {"NOUN", "case", "number", "animacy", "person", "gender"}
        elif tag.POS in ["ADJF"]:
            keys = {"ADJF", "case", "gender", "number"}
        elif tag.POS == "ADJS":
            keys = {"ADJS", "gender", "number"}
        elif tag.POS == "VERB":
            keys = {"mood", "tense", "aspect", "person", "voice", "gender", "number"}
        elif tag.POS == "INFN":
            keys = {"INFN", "aspect"}
        elif tag.POS in ["PRTF"]:
            keys = {"PRTF", "case", "gender", "number", "voice", "aspect"}
        elif tag.POS == "PRTS":
            keys = {"PRTS", "gender", "number", "voice", "aspect"}
        elif tag.POS == "GRND":
            keys = {"GRND", "voice", "aspect"}
        elif tag.POS == "COMP":
            keys = {"COMP"}
        else:
            keys = {}
        values = {(getattr(tag, key) if key.islower() else key) for key in keys}
        return {x for x in values if x is not None}


class EnInflector():
    """
    """

    def __init__(self, classical_pluralize: bool=True):
        self.classical_pluralize = classical_pluralize

    
    def _nltk_postag_to_pattern_postag(self, pos_tag: str):
        if pos_tag[1].startswith('N'):
            return en.NOUN
        elif pos_tag[1].startswith('V'):
            return en.VERB
        elif pos_tag[1].startswith('J'):
            return en.ADJECTIVE
        elif pos_tag[1].startswith('R'):
            return en.ADVERB

    def get_lemma_form(self, token: str):
        return en.lemma(token)

    
    def get_morpho_tag(self, token: str, pos_tag):
        morpho_tag = {}
        if pos_tag[1].startswith('N'):
            morpho_tag.update({'pos_tag': pos_tag, 'plur': pos_tag[1].endswith('S')})
        elif pos_tag[1].startswith('V'):
            morpho_tag.update({'pos_tag': pos_tag, 'tense': en.tenses(token)[0]})
        elif pos_tag[1].startswith('J'):
            morpho_tag.update({'pos_tag': pos_tag, 'comp': pos_tag[1].endswith('R'), 'supr': pos_tag[1].endswith('S')})
        elif pos_tag[1].startswith('R'):
            morpho_tag.update({'pos_tag': pos_tag, 'comp': pos_tag[1].endswith('R'), 'supr': pos_tag[1].endswith('S')})
        return morpho_tag
    
    def inflect_token(self, token, morpho_tag):
        if morpho_tag['tense']:
            return en.conjugate(token, morpho_tag['tense'])
        if morpho_tag['plur']:
            return en.inflect.pluralize(token,
                                        pos=self._nltk_postag_to_pattern_postag(morpho_tag['pos_tag']),
                                        classical=self.classical_pluralize)
        if morpho_tag['comp']:
            return en.inflect.comparative(token)
        if morpho_tag['supr']:
            return en.inflect.superlative(token)


class RuThesaurus():
    """
    """

    def __init__(self, dir_path, with_source_token: bool=False):
        self.dir_path = Path(dir_path) 
        self.with_source_token = with_source_token
        self.entry_root = ET.parse(self.dir_path / "text_entry.xml").getroot()
        self.synonyms_root = ET.parse(self.dir_path / "synonyms.xml").getroot()
        self.test_time = []
        print(self.test_time)

    
    def _find_synonyms(self, lemma: str) -> List[str]:
        start = time.time() #TEST

        lemma = lemma.upper()
        #1 
        entry_id_set = set(map(lambda x: x.get('id'), self.entry_root.findall(f"./entry[lemma='{lemma}']")))
        #2
        concept_id_set = set()
        for entry_id in entry_id_set:
            concept_id_set.update(set(map(lambda x: x.get('concept_id'), self.synonyms_root.findall(f"./entry_rel[@entry_id='{entry_id}']"))))
        #3
        syn_entry_id_set = set()
        for concept_id in concept_id_set:
            syn_entry_id_set.update(set(map(lambda x: x.get('entry_id'), self.synonyms_root.findall(f"./entry_rel[@concept_id='{concept_id}']"))))
        #4
        synlist = list()
        for syn_entry_id in syn_entry_id_set:
            synlist += list(map(lambda x: x.text, self.entry_root.findall(f"./entry[@id='{syn_entry_id}']/lemma")))
        
        end = time.time() #TEST
        self.test_time.append(end-start) #TEST

        return synlist


    def _filter(self, synlist: List[str], source_lemma: str) -> List[str]:
        filtered_syn = set(filter(lambda x: len(x.split()) == 1, synlist))
        if self.with_source_token:
            filtered_syn.update([source_lemma])
        else:
            filtered_syn.discard([source_lemma])
        return list(filtered_syn)

    
    def get_syn(self, lemma: str, pos_tag: str=None) -> List[str]:
        lemma = lemma.upper()
        synonyms = self._find_synonyms(lemma)
        synonyms = self._filter(synonyms, lemma)
        return synonyms
