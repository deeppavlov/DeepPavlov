import xml.etree.ElementTree as ET
import pymorphy2
from deeppavlov.core.data.utils import download_decompress
from pathlib import Path
from typing import List

from deeppavlov.core.data.utils import download_decompress
from deeppavlov.core.common.log import get_logger

logger = get_logger(__name__)

#прикрутить морфотегер от Алексея
#он возвращает значения в universal... есть библиотека russian_tagsets которая переводит pymorphy тэги в universal...
class RuWordnet:
        
    def __init__(self, dir_path: str, with_source_token: bool=False):
        url = "https://github.com/SultanovAR/___ruthes-lite2_download/archive/master.zip"    
        url = 'http://files.deeppavlov.ai/datasets/ruthes-lite2.tar.gz'

        dir_path = Path(expand_path(dir_path))
        required_files = ['text_entry.xml', 'synonyms.xml']
        if not dir_path.exists()
            dir_path.mkdir()
        
        if not all((dir_path/f)/exists() for f in required_files):
            download_decompress(url, dir_path)

        self.with_source_token = with_source_token
        self.entry_root = ET.parse(next(self.ruthes_path.rglob('text_entry.xml'))).getroot()
        self.synonyms_root = ET.parse(next(self.ruthes_path.rglob('synonyms.xml'))).getroot()
        self.morph = pymorphy2.MorphAnalyzer()
    
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

    def _find_synonyms(self, lemma: str) -> List[str]:
        #parsing ruthes-lite-2 file
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
        return synlist
    
    def _filter(synlist: List[str], init_form: pymorphy2.analyzer.Parse, source_token: str) -> List[str]:
        init_form = self._extract_morpho_requirements(init_form)
        synset = set()
        for syn in filter(lambda x: len(x.split())) == 1, synlist):
            inflected_syn = self.morph.parse(syn)[0].inflect(init_form)
            if inflected_syn:
                if not(self.with_source_token) and inflected_syn != source_token:
                    synlist.update([inflected_syn.word])
                elif self.with_source_token:
                    synlist.update([inflected_syn.word])
        return list(synset)

    #def _find_filtered_synonyms(self, morphem):
    #    #finds synonyms, filters them(len=1) and tries to infect it to right form
    #    lemma = morphem.normal_form
    #    init_form = self._extract_morpho_requirements(morphem.tag)
    #    synlist = set()
    #    for syn in filter(lambda x: len(x.split()) == 1, self._find_synonyms(lemma.upper())):
    #        syn_inflected = self.morph.parse(syn)[0].inflect(init_form)
    #        if syn_inflected:
    #            if syn_inflected.word != morphem.word: #synlist without source token
    #                synlist.update([syn_inflected.word])
    #    return list(synlist) if synlist else None

    def get_synlist(self, token: str, pos_tag: str=None) -> List[str]:
        if token:
            morphem = self.morph.parse(x)[0]
            synonyms = self._find_synonyms(morphem.normal_form)
            synonyms = self._filter(synonyms, morphem.tag, morphem.normal_form)
            return synonyms

    #def get_synset(self, prep_tokens: List[str], source_pos_tag: List[str]=None) -> List[List[str]]:
    #    """Generating list of synonyms for each token in prep_tokens, that isn't equal to None
    #        Args:
    #            prep_tokens: preprocessed soucrce_tokens, where all tokens that do not need to search for synonyms are replaced by None
    #            source_pos_tag: pos tags for source sentence, in nltk.pos_texitag format
    #        Return:
    #            List of list of synonyms without source token, for tokens for which no synonyms were found, return None
    #    """
    #    morphems = list(map(lambda x: self.morph.parse(x)[0] if x else None, prep_tokens))
#
    #    result = list(map(lambda x: self._find_filtered_synonyms(x) if x else None, morphems))
    #    return result

if __name__ == "__main__":
    print("""Testing the module "RuSyn" """)
    import shutil
    shutil.rmtree(Path(expand_path('./ruthes')), ignore_errors=True)

    #1 download data
    ru = RuSyn('./ruthes')

    #2 Read from existing dir
    ru = RuSyn('./ruthes')

    #3 checking synset for lemma "РАБ"
    # The right synset were taken from 
    # http://www.labinform.ru/pub/ruthes/te/17/001/172506.htm
    assert set(['НЕВОЛЬНИК', 'НЕВОЛЬНИЦА', 'НЕВОЛЬНИЧИЙ', 'РАБ', 'РАБСКИЙ', 'РАБЫНЯ']) == set(ru._find_synonyms('РАБ'))

    #4 checking synset for lemma "ЩЕБЕНЬ"
    assert set(['ЩЕБЕНКА', 'ЩЕБЕНОЧНЫЙ', 'ЩЕБЕНЬ', 'ЩЕБНЕВОЙ', 'ЩЕБНЕВЫЙ']) == set(ru._find_synonyms('ЩЕБЕНЬ'))

    #5 checking result synset for "прочитана"
    assert set(['дочитана', 'прочтена']) == set(ru.get_synset(['прочитана'], ['прочитана'])[0])
    # example of using
    #print(5, ru._find_synonyms('СЦЕНА'))  
    #print(ru.get_synset(['раб', 'щебнем', 'три'], ['раб', 'щебнем', 'три']))
    #print(ru.get_synset(['прочитанная', 'прочитана', 'прочитав'], ['прочитанная', 'прочитана', 'прочитав']))
    #print(ru.get_synset(['лучше', 'год', 'слово'], ['лучше', 'год', 'слово']))
    #print(ru.get_synset(['преподавая', 'славистику', 'русскому'], ['преподавая', 'славистику', 'русскому']))
    #print(ru.get_synset(['сцены', 'конкретного', 'тривиальность'], ['сцены', 'конкретного', 'тривиальность']))
    #print(ru.get_synset(['сексуальную', 'распущенность', 'надрыв'], ['сексуальную', 'распущенность', 'надрыв']))
    #print(ru.get_synset(['чувства', 'сумасшедшем', 'кротким'], ['чувства', 'сумасшедшем', 'кротким']))
    #print(ru.get_synset(['определено', 'передает', 'писал'], ['определено', 'передает', 'писал']))
    #print(ru.get_synset(['быть', 'бытие', 'означает'], ['быть', 'бытие', 'означает']))