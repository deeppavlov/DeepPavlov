import xml.etree.ElementTree as ET
from pymorphy2 import MorphAnalyzer

from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.data.utils import download_decompress, mark_done, is_done

from pathlib import Path
from typing import List

class RuSynWordnet:

    def _download_data(self, data_path: str) -> None:
        """download ruthes-lite2 thesaurus to data_path"""
        url = "https://github.com/SultanovAR/___ruthes-lite2_download/archive/master.zip"
        download_decompress(url, data_path)
        mark_done(data_path)      

    def __init__(self, ruthes_path: str):
        self.ruthes_path = Path(expand_path(ruthes_path))

        if not is_done(self.ruthes_path):
            self._download_data(self.ruthes_path)

        if self.ruthes_path.is_dir():
            self._entry = ET.parse(next(self.ruthes_path.rglob('text_entry.xml')))
            self._synonyms = ET.parse(next(self.ruthes_path.rglob('synonyms.xml')))
            self.entry_root = self._entry.getroot()
            self.synonyms_root = self._synonyms.getroot()
            
        self.morph = MorphAnalyzer()
    
    def _extract_morpho_requirements(self, tag):
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
        synset = list()
        for syn_entry_id in syn_entry_id_set:
            synset += list(map(lambda x: x.text, self.entry_root.findall(f"./entry[@id='{syn_entry_id}']/lemma")))
        return synset

    def _find_filtered_synonyms(self, morphem):
        #finds synonyms, filters them(len=1) and tries to infect it to right form
        lemma = morphem.normal_form
        init_form = self._extract_morpho_requirements(morphem.tag)
        synlist = set()
        for syn in filter(lambda x: len(x.split()) == 1, self._find_synonyms(lemma.upper())):
            syn_inflected = self.morph.parse(syn)[0].inflect(init_form)
            if syn_inflected:
                if syn_inflected.word != morphem.word: #synlist without source token
                    synlist.update([syn_inflected.word])
        return list(synlist) if synlist else None

    def get_synset(self, prep_tokens: List[str], source_pos_tag: List[str]=None) -> List[List[str]]:
        """Generating list of synonyms for each token in prep_tokens, that isn't equal to None
            Args:
                prep_tokens: preprocessed soucrce_tokens, where all tokens that do not need to search for synonyms are replaced by None
                source_pos_tag: pos tags for source sentence, in nltk.pos_texitag format
            Return:
                List of list of synonyms without source token, for tokens for which no synonyms were found, return None
        """
        # token </s> to do
        morphems = list(map(lambda x: self.morph.parse(x)[0] if x else None, prep_tokens))
        result = list(map(lambda x: self._find_filtered_synonyms(x) if x else None, morphems))
        return result

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