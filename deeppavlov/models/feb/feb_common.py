
from enum import Enum

class Utterance(Enum):
    VALUE = 'utt_str'
    TYPE = 'query_name'
    NAMED_ENTITY_LST = 'nent_lst'
    ERROR = 'utt_error'
    ERROR_VAL_LST = 'utt_error_val'
    RESULT = 'result'

class UtteranceErrors(Enum):
    QID_NOT_FOUND = 'qid_not_found'
    WIKIDATA_QUERY_ERROR = 'wikidata_query_error'

class NamedEntity(Enum):
    NE_STRING = 'nent_str'
    NE_TYPE = 'nent_type'
    NE_QID = 'nent_qid'

class NamedEntityType(Enum):
    AUTHOR = 'author'
    BOOK = 'book'