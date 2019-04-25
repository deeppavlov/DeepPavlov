from typing import List
from nltk import pos_tag as nltk_pos_tagger
from numpy.random import sample
from itertools import repeat


class WordFilter:
    """Class that decides which tokens should not be replaced
    Args:
        replace_freq: [0, 1] propability of token that passed thought other filters to be replaced
        isalpha_only: filter based on string method 'isalpha'
        not_replaced_tokens: List of tokens that shouldn't be replaced
    Attributes:
        replace_freq: [0, 1] propability of token that passed thought other filters to be replaced
        isalpha_only: filter based on string method 'isalpha'
        not_replaced_tokens: List of tokens that shouldn't be replaced
    """

    def __init__(self,
                 replace_freq: float,
                 isalpha_only: bool,
                 not_replaced_tokens: List[str]):
        self.replace_freq = replace_freq
        self.isalpha_only = isalpha_only
        self.not_replaced_tokens = not_replaced_tokens

    def filter_isalpha_only(self, tokens):
        if self.isalpha_only:
            return map(lambda x: x.isalpha(), tokens)
        else:
            return repeat(True, len(tokens))

    def filter_not_replaced_token(self, tokens):
        return map(lambda x: not x in self.not_replaced_tokens, tokens)

    def filter_based_on_pos_tag(self, tokens, pos_tags):
        pass

    def filter_frequence(self, prev_decision):
        return map(lambda x: sample() < self.replace_freq if x else x, prev_decision)

    def filter_united(self, tokens, pos_tags):
        return map(lambda x, y, z: all([x,y,z]),
                   self.filter_based_on_pos_tag(tokens, pos_tags),
                   self.filter_not_replaced_token(tokens),
                   self.filter_isalpha_only(tokens))

    def __call__(self, tokens, pos_tags):
        """It filters tokens based on replace_freq, isalpha_only, not_replaced_token and pos_tags of tokens
        Args:
            tokens: tokens that will be filtered
        Return:
            List of boolean values
        """
        filtered = self.filter_united(tokens, pos_tags)
        return self.filter_frequence(filtered)


class EnWordFilter(WordFilter):
    """Class that decides which tokens should not be replaced, for english language
    Args:
        replace_freq: [0, 1] propability of token that passed thought other filters to be replaced
        isalpha_only: filter based on string method 'isalpha'
        not_replaced_tokens: List of tokens that shouldn't be replaced
        replaced_pos_tags: List of pos_tags that can be replaced, e.g. 'n' for Noun, 'v' for Verb, 'a' for Adjective, 'r' for Adverb
        is_replace_proper_noun: to replace proper noun or not, based on pos_tag
        is_replace_modal_verb: to replace modal verb or not, based on pos_tag
    Attributes:
        replace_freq: [0, 1] propability of token that passed thought other filters to be replaced
        isalpha_only: filter based on string method 'isalpha'
        not_replaced_tokens: List of tokens that shouldn't be replaced
        replaced_pos_tags: List of pos_tags that can be replaced, e.g. 'n' for Noun, 'v' for Verb, 'a' for Adjective, 'r' for Adverb
        is_replace_proper_noun: to replace proper noun or not, based on pos_tag
        is_replace_modal_verb: to replace modal verb or not, based on pos_tag
    """

    def __init__(self,
                 replace_freq: float,
                 isalpha_only: bool,
                 not_replaced_tokens: List[str], # = ["n't"],
                 replaced_pos_tags: List[str],
                 is_replace_proper_noun: bool,
                 is_replace_modal_verb):
        super(EnWordFilter, self).__init__(replace_freq, isalpha_only, not_replaced_tokens)
        self.replaced_pos_tags = replaced_pos_tags
        self.postag_to_nltk_postag = {
            'n': ['NN', 'NNS'],
            'v': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'a': ['JJ', 'JJR', 'JJS'],
            'r': ['RB', 'RBR', 'RBS', 'PDT']
        }
        if is_replace_proper_noun:
            self.postag_to_nltk_postag['n'].extend(['NNP', 'NNPS'])
        if is_replace_modal_verb:
            self.postag_to_nltk_postag['v'].extend(['MD'])
        self._sentence_pos_tag = []

    def get_nltk_replaced_postags(self, tag_list):
        return sum(map(lambda x: self.postag_to_nltk_postag[x], tag_list),[])

    def filter_based_on_pos_tag(self, tokens, pos_tags):
        """Function that filters tokens with pos_tags and rules
        Args:
            tokens: tokens that will be filtered
            pos_tags: pos_tags for 'tokens', in nltk.pos_tag format
        Return:
            List of boolean values
        """
        replaced_nltk_postags = self.get_nltk_replaced_postags(self.replaced_pos_tags)
        result = []
        prev_postag_is_EX = False
        for token, pos_tag in self._sentence_pos_tag:
            if pos_tag == 'EX':
                prev_postag_is_EX = True
                result.append(False)
            elif prev_postag_is_EX and pos_tag.startswith('V'):
                prev_postag_is_EX = False
                result.append(False)
            else:
                result.append(pos_tag in replaced_nltk_postags)
        return result


class RuWordFilter(WordFilter):
    """

    """

    def __init__(self,
                 replace_freq: float,
                 isalpha_only: bool,
                 not_replaced_tokens: List[str],
                 replaced_pos_tags: List[str],
                 is_use_numeral_adjective):
        super(RuWordFilter, self).__init__(replace_freq, isalpha_only, not_replaced_tokens)
        self.replaced_pos_tags = replaced_pos_tags
        self.postag_to_nltk_postag = {
            's': ['S'],
            'num': ['NUM'],
            'v': ['V'],
            'a': ['A'],
            'r': ['ADV']
        }
        if is_use_numeral_adjective:
            self.postag_to_nltk_postag['a'].append('ANUM')

    def get_nltk_replaced_postags(self, tag_list):
        return sum(map(lambda x: self.postag_to_nltk_postag[x], tag_list), [])

    def filter_based_on_pos_tag(self, tokens, pos_tags):
        replaced_nltk_postags = self.get_nltk_replaced_postags(self.replaced_pos_tags)
        print(pos_tags)
        return map(lambda x: x[1] in replaced_nltk_postags, pos_tags)

if __name__ == '__main__':
    #a = EnWordFilter(1, True, ['suka'], ['n', 'v', 'a', 'r'], True, True)
    a = RuWordFilter(1, False, ['Илья'], ['s', 'num', 'a', 'r', 'v'], True)
    #print(list(a.filter_frequence([True]*100)))
    test_str = "Илья оторопел и дважды перечитал бумажку ."
    posss = nltk_pos_tagger(test_str.split(), lang='rus')
    result = list(a(test_str.split(), posss))
    print(*list(zip(posss, result)), sep='\n')