
from razdel import sentenize, tokenize
import re

from question2wikidata.server_queries import queries



def var_dump(msg, header = ''): print(f'-----{header}-----\n{msg}\n----\n')
def get_category(x): 
    cats = ['book_author',
        'book_written',
        'book_published',
        'book_characters',
        'book_genre',
        'book_main_theme',
        'author_birthplace',
        'author_productions',
        'author_genres',
        'author_when_born',
        'author_where_lived',
        'author_languages',
        'author_when_died',
        'author_where_died',
        'author_where_buried',
        'author_inspired_by']
    for cat in cats: 
        if cat in x:
            return cat
    else:
        return None

class FebError(object):

    ET = 'et_' # error type
    ET_SYS = 'et_sys_' # system error
    ET_LOG = 'et_log_' # busyness logic error
    ET_INP_DATA = 'et_log_inpdata_' # input data check error

    EC = 'ec_' # error cause
    EC_DATA = EC + 'data_'
    EC_DATA_LACK = EC_DATA + 'lack_' # insufficient data
    EC_DATA_NONE = EC_DATA_LACK + 'none_' # no value at all
    EC_DATA_DISABIG = EC_DATA + 'disambiguation_' # there are two or more variants
    EC_DATA_VAL = EC_DATA + 'val_' # wrong data value
    EC_DATA_TYPE = EC_DATA + 'type_' # wrong data type
    EC_EXCEPTION = EC + 'exception_'
    EC_WRONG_TYPE = EC + 'wr'

    @staticmethod
    def is_err_subtype(err_code_1, err_code_2):
        """
        Check err_code_1 is special case (more detailed, i.e. longer) of err_code_2
        """
        return err_code_2.find(err_code_1, 0, len(err_code_1)) == 0


    def __init__(self, error_type, component, cause_d):
        """

        :param error_type: ET constant
        :param component: component where error occur
        :param cause_d: dict of error causes and its context
        """
        super().__init__()
        self.type = error_type
        self.component_name = type(component).__name__
        self.cause_d = cause_d

    def __repr__(self):
        vals = ', '.join(f'{k}={repr(v)}' for k, v in self.__dict__.items())
        return f'{self.__class__.__name__}({vals})'





class FebObject(object):

    def __init__(self, **kwargs):
        super().__init__()
        self.type = None  # object type
        self.errors = [] # errors list

    def add_error(self, error):
        print(f'new error {error}')
        self.errors.append(error)

    def __repr__(self):
        vals = ', '.join(f'{k}={repr(v)}' for k, v in self.__dict__.items())
        return f'{self.__class__.__name__}({vals})'

    def has_errors(self):
        return len(self.errors) != 0

    @classmethod 
    def recursive_json(cls, obj):
        if isinstance(obj, list) or isinstance(obj, tuple) or isinstance(obj, set):
            # print(obj, type(obj))
            return [FebObject.recursive_json(element) for element in obj]
        elif isinstance(obj, dict):
            # print(obj, type(obj))
            return {k: FebObject.recursive_json(v) for k, v in obj.items()}
        elif isinstance(obj, FebObject) or isinstance(obj, FebError):
            # print(obj, type(obj))
            return {k: FebObject.recursive_json(v) for k,v in obj.__dict__.items()}
        else:
            return obj



class FebToken(FebObject):
    # Token types:
    PUNCTUATION = 't_punktuation'
    NOT_PUNCTUATION = 't_text'
    WORD = 't_word'
    NUMBER = 't_number'
    OTHER = 't_other'

    # Token language types:
    WORD_LANGUAGE_RU = 't_word_lng_ru'
    WORD_LANGUAGE_OTHER = 't_word_lng_other'

    # Token tags:
    TAG_EOFS = 'tag_eofs'
    TAG_AUTHOR = 'tag_author'
    TAG_BOOK = 'tag_book'   




    @staticmethod
    def sentenize(text):
        """
        Split text into sentences
        :param text:
        :return: list [(start, stop, text), ... ]
        """
        return list(sentenize(text))

    @staticmethod
    def tokenize(text, flat=True, tag_eofs=True):
        sent_l = FebToken.sentenize(text)
        sent_tok_ll = [(list(map(lambda t: FebToken(sent.start + t.start, sent.start + t.stop,
                                                    t.text, source_text=text),
                                 tokenize(sent.text)))) for sent in sent_l]
        if tag_eofs:
            for tok_l in sent_tok_ll:
                if len(tok_l) > 0:
                    tok_l[-1].tags.add(FebToken.TAG_EOFS)
        if flat:
            return [tok for tok_l in sent_tok_ll for tok in tok_l]
        else:
            return sent_tok_ll


    @staticmethod
    def stemmer(sentence):
        from pymystem3 import Mystem
        STEMMER = Mystem()
        pos_map = {
                'A': 'ADJ',
                'ADV': 'ADV',
                'ADVPRO': 'ADV',
                'ANUM': 'ADJ',
                'APRO': 'DET',
                'COM': 'ADJ',
                'CONJ': 'SCONJ',
                'INTJ': 'INTJ',
                'NONLEX': 'X',
                'NUM': 'NUM',
                'PART': 'PART',
                'PR': 'ADP',
                'S': 'NOUN',
                'SPRO': 'PRON',
                'UNKN': 'X',
                'V': 'VERB'
            }
        processed = STEMMER.analyze(sentence)
        tagged = []
        for w in processed:
            try:
                lemma = w["analysis"][0]["lex"].lower().strip()  
                text = w['text']
                pos = w["analysis"][0]["gr"].split(',')[0]
                pos = pos.split('=')[0].strip()
                pos = pos_map.get(pos, 'X')
                start = sentence.index(text)
                stop = start + len(text)
                token = FebToken(0, 0, text, normal_form = lemma, pos=pos)
                tagged.append(token)
            except (KeyError, IndexError):
                continue
        return tagged



    def __init__(self, start, stop, text, **kwargs):
        """

        :param start:
        :param stop:
        :param text:
        :param kwargs: source_text, t_type, lang
        """
        super().__init__()

        self.start = start
        self.stop = stop
        self.text = text
        self.source_text = kwargs.get('ttype', None)  # source text string
        if self.text:
            self.set_t_type()
        self.lang = kwargs.get('lang', None)  # token language
        self.normal_form = kwargs.get('normal_form', None)
        self.pos = kwargs.get('pos', None)
        self.tags = set()  # tokens tags (example: markers of NER)

    # TODO: identify other types
    def set_t_type(self, only_punktuation=True):
        assert only_punktuation, 'Other options not implemented!'
        if self.text:
            if re.match('.*[\w\d_].*', self.text) is None:  # TODO: check regexp
                self.type = FebToken.PUNCTUATION
            else:
                self.type = FebToken.NOT_PUNCTUATION
        else:
            raise ValueError('text value is not set')

    def set_pos(self, pos):
        self.pos = pos
    def set_normal_form(self, normal_form):
        self.normal_form = normal_form        

    def __repr__(self):
        vals = str(self)
        return f'{self.__class__.__name__}({vals})'

    def __str__(self):
        rs = f'({self.start}, {self.stop}, {self.text}'
        if self.type:
            rs += f', type={self.type}'
        if self.lang:
            rs += f', lang={self.lang}'
        if self.tags:
            rs += f', tags={self.tags}'
        if self.pos:
            rs += f', pos={self.pos}' 
        if self.normal_form:
            rs += f', normal_form={self.normal_form}'                       
        rs += ')'
        return rs
    def __eq__(self, other):
        return self.text == other.text



class FebEntity(FebObject):
    # class attributes:
    # types:
    AUTHOR = 'author'
    BOOK = 'book'

    def __init__(self, type,  **kwargs):
        super().__init__(**kwargs)
        self.type = type
        self.tokens = kwargs.get('tokens', None) # list of tokens
        self.qid = kwargs.get('qid', None) # id in Wikidata
        self.qname = kwargs.get('qname', None) # name in Wikidata
        self.normal_form = kwargs.get('normal_form', None)

    def tokens_to_search_string(self):
        return ' '.join(t.text for t in self.tokens)


class FebAuthor(FebEntity):

    def __init__(self,  **kwargs):
        super().__init__(FebEntity.AUTHOR, **kwargs)


class FebBook(FebEntity):

    def __init__(self,  **kwargs):
        super().__init__(FebEntity.BOOK, **kwargs)

class FebIntent(FebObject):
    """
    'book_author'
'book_written'
'book_published'
'book_characters'
'book_genre'
'book_main_theme'

'author_birthplace'
'author_productions'
'author_genres'
'author_when_born'
'author_where_lived'
'author_languages'
'author_when_died'
'author_where_died'
'author_where_buried'
'author_inspired_by'

    """
    supported_types = {q for q in queries.keys() if q[:5] != 'help_'}
    UNSUPPORTED_TYPE = 'unsupported_type'

    @classmethod
    def in_supported_types(cls, type):
        return type in cls.supported_types

    def __init__(self, type,  **kwargs):
        super().__init__(**kwargs)

        self.type = type
        self.confidence = kwargs.get('confidence', None) # float confidence level

        self.result_qid = kwargs.get('result_qid', None) # result id in Wikidata
        self.result_str = kwargs.get('result_str', None) # result string type


class FebUtterance(FebObject):

    def __init__(self, text, **kwargs):
        super().__init__(**kwargs)

        self.text = text # input text
        self.tokens = None # list of tokens
        self.entities = [] # list of entities
        self.intents = [] # list of intents
        self.re_text = None # responce text

    def to_dump(self):
        # return {k: [FebObject.recursive_json(item) for item in v if isinstance(item, (FebObject, FebError)) ] for k, v in self.__dict__.items() if v is not None }
        return {k: FebObject.recursive_json(v) for k, v in self.__dict__.items() if v is not None}




    def return_text(self):
        if self.re_text:
            return self.re_text
        else:
            return 'Что-то пошло не так, попробуйте еще раз.'


