from nltk.corpus import wordnet as wn
from nltk import pos_tag
import pattern.en as en
from pattern.en import ADJECTIVE, NOUN, VERB, ADVERB
from itertools import repeat
import nltk

from typing import List, Tuple
from collections import defaultdict

class EnSynWordnet:
    """Finding synonyms for each token in given list, for english language

    Args:
        classical_pluralize: whether to use classical pluralize method in pattern.en.pluralize
    Attributes:
        classical_pluralize: whether to use classical pluralize method in pattern.en.pluralize
        wn_tag_to_pattern:   vocabulary between pos_tags in wordnet and pos_tags in pattern
    """

    def __init__(self, classical_pluralize: bool=True):
        self.classical_pluralize = classical_pluralize
        self.wn_tag_to_pattern = {wn.VERB: VERB, wn.NOUN: NOUN, wn.ADJ: ADJECTIVE, wn.ADV: ADVERB}

    def _download_data(self):
        """download all neccesary data"""
        nltk.download('wordnet')

    def _find_synonyms(self, token_morph: Tuple[dict, str]):
        if token_morph:
            synonyms = set()
            if token_morph[0]['pos'] == wn.ADJ: # take all Adjective satellites
                for synset in wn.synsets(token_morph[1], pos='s'):
                    for synset_lemma in synset.lemmas():
                        if synset_lemma.name() != token_morph[1]:#take all lemmas that not equal to source token
                            synonyms.update([synset_lemma.name()])
            if token_morph[0]['pos']:# if there are synset for this pos_tag
                for synset in wn.synsets(token_morph[1], pos=token_morph[0]['pos']):
                        for synset_lemma in synset.lemmas():
                            if synset_lemma.name() != token_morph[1]:
                                synonyms.update([synset_lemma.name()])
            return list(synonyms)

    def _lemmatize(self, token: str, pos_tag: Tuple[str, str]) -> List[str]:
        if token:
            init_form = defaultdict(bool)
            lemma = token
            if pos_tag[1].startswith('N'):
                init_form.update({'pos': wn.NOUN, 'plur': pos_tag[1].endswith('S')})
                lemma = en.inflect.singularize(token)
            elif pos_tag[1].startswith('V'):
                init_form.update({'pos': wn.VERB,\
                                    'tense': en.tenses(token)[0]})
                lemma = en.lemma(token)
            elif pos_tag[1].startswith('J'):
                init_form.update({'pos': wn.ADJ, 'comp': pos_tag[1].endswith('R'), 'supr': pos_tag[1].endswith('S'),\
                                    'plur': en.inflect.pluralize(token, pos=ADJECTIVE, classical=self.classical_pluralize) == token})
            elif pos_tag[1].startswith('R'):
                init_form.update({'pos': wn.ADV, 'comp': pos_tag[1].endswith('R'),\
                                    'supr': pos_tag[1].endswith('S')})
                
            return init_form, lemma
    
    def _inflect_to_init_form(self, token, init_form):
        token = token.split('_')
        if init_form['plur']:
            token[0] = en.inflect.pluralize(token[0], pos=self.wn_tag_to_pattern[init_form['pos']], classical=self.classical_pluralize)
        if init_form['tense']:
            token[0] = en.conjugate(token[0], init_form['tense'])
        if init_form['comp']:
            token[0] = en.inflect.comparative(token[0])
            if len(token[0].split()) == 2:# if added 'more' or 'most' then need delete it
                token[0] = token[0].split()[1]
        if init_form['supr']:
            token[0] = en.inflect.superlative(token[0])
            if len(token[0].split()) == 2:# if added 'more' or 'most' then need delete it
                token[0] = token[0].split()[1]
        return " ".join(token)
        
    def _unlemmatize(self, synonyms_for_one_token, token_morph):
        if synonyms_for_one_token:
            synset = set(map(self._inflect_to_init_form, synonyms_for_one_token, repeat(token_morph[0], len(synonyms_for_one_token))))
            return list(synset)

    def get_synset(self, prep_tokens: List[str], source_pos_tag: List[str]=None) -> List[List[str]]:
        """Generating list of synonyms for each token in prep_tokens, that isn't equal to None
            Args:
                prep_tokens: preprocessed source tokens, where all tokens that do not need to search for synonyms are replaced by None
                source_pos_tag: pos tags for source sentence, in nltk.pos_tag format
            Return:
                List of list of synonyms without source token, for tokens for which no synonyms were found, return None
        """
        tokens_morph = list(map(self._lemmatize, prep_tokens, source_pos_tag))
        #find synonyms excluding the source word
        list_synonyms = list(map(self._find_synonyms, tokens_morph))
        #inflect and return synonyms
        return list(map(self._unlemmatize, list_synonyms, tokens_morph))

if __name__ == '__main__':
    from nltk import word_tokenize
    ensyn_test = EnSynWordnet()
    #test: Finding the correct list of synonoms for noun,
    # the reference list was taken from the site
    # http://wordnetweb.princeton.edu/perl/webwn?s=eat&sub=Search+WordNet&o2=&o0=1&o8=1&o1=1&o7=&o5=&o9=&o6=&o3=&o4=&h=00000000
    test_sentence = 'frog'
    test_sentence = word_tokenize(test_sentence)
    pos_test_sentece = pos_tag(test_sentence)
    res = list(map(lambda x: x.lower(), ensyn_test.get_synset(test_sentence, test_sentence, pos_test_sentece)[0]))
    assert set(res) ==  set(['toad', 'toad frog', 'anuran', 'batrachian', 'salientian', 'gaul'])

    #test: inflect of noun
    test_sentence = 'frogs'
    test_sentence = word_tokenize(test_sentence)
    pos_test_sentece = pos_tag(test_sentence)
    res = list(map(lambda x: x.lower(), ensyn_test.get_synset(test_sentence, test_sentence, pos_test_sentece)[0]))
    assert set(res) == set(['toads', 'toads frog', 'anurans', 'batrachians', 'salientians', 'gauls'])
    
    #test: elements in prep list that marked by None don't proceced
    test_sentence = 'frog frogs'
    test_sentence = word_tokenize(test_sentence)
    pos_test_sentece = pos_tag(test_sentence)
    res = ensyn_test.get_synset(test_sentence, [None, None], pos_test_sentece)
    assert set(res) ==  set([None, None])

    #inflect verb
    """
    module pattern.en can't inflect feed to fed, so test passing 'conditionaly'
    """
    test_sentence = 'you ate'
    test_sentence = word_tokenize(test_sentence)
    pos_test_sentece = pos_tag(test_sentence)
    res = ensyn_test.get_synset(test_sentence, [None, 'ate'], pos_test_sentece)[1]
    res = list(map(lambda x: x.lower(), res))
    ref = ['feed', 'ate on', 'consumed',\
             'ate up', 'used up', 'depleted',\
            'exhausted', 'ran through', 'wiped out',\
            'corroded', 'rusted']
    assert set(res) == set(ref)
 
    #return None when synonyms don't found
    test_sentence = "you're fluffest"
    test_sentence = word_tokenize(test_sentence)
    pos_test_sentece = pos_tag(test_sentence)
    res = ensyn_test.get_synset(test_sentence, [None, None, 'fluffest'], pos_test_sentece)
    ref = [None, None, None]
    assert res == ref
