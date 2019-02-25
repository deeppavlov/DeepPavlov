
from enum import Enum


from overrides import overrides

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.common.log import get_logger

from .feb_objects import *

from question2wikidata.questions import pretty_json

log = get_logger(__name__)

# class Utterance(Enum):
#     VALUE = 'utt_str'
#     TYPE = 'query_name'
#     NAMED_ENTITY_LST = 'nent_lst'
#     ERROR = 'utt_error'
#     ERROR_VAL_LST = 'utt_error_val'
#     RESULT = 'result'
#
# class UtteranceErrors(Enum):
#     QID_NOT_FOUND = 'qid_not_found'
#     WIKIDATA_QUERY_ERROR = 'wikidata_query_error'
#
# class NamedEntity(Enum):
#     NE_STRING = 'nent_str'
#     NE_TYPE = 'nent_type'
#     NE_QID = 'nent_qid'
#
# class NamedEntityType(Enum):
#     AUTHOR = 'author'
#     BOOK = 'book'


# @register('feb_t1_parser')
class FebComponent(Component):
    """Convert batch of strings
    sl = ["author_birthplace author Лев Николаевич Толстой",
      -(to)->
        utterence object
      """
    START_COMPONENT = 'start_component'
    INTERMEDIATE_COMPONENT = 'intermediate_component'
    FINAL_COMPONENT = 'final_component'

    @classmethod
    def component_type(cls):
        # TODO: abstract
        assert False, 'need to override'
        return None

    def __init__(self, **kwargs):
        super().__init__()

    @overrides
    def __call__(self, batch, *args, **kwargs):
        res_batch = []
        for in_obj in batch:

            try:  # Error isolation between objects of batch
                if self.component_type() == self.START_COMPONENT:
                    if isinstance(in_obj, str):
                        utt = FebUtterance(in_obj)
                    else:
                        utt = FebUtterance('')
                        # utt.add_error(FebError(FebError.ET_SYS, self, {FebError.EC_WRONG_TYPE: in_obj}))
                        raise TypeError(f"start_component is not implemented for `{type(in_obj)}`")
                else:
                    utt = in_obj
                var_dump(in_obj)
                obj_l = self.test_and_prepare(utt)
                ret_obj_l = []
                # print(f'P1: obj_l={obj_l}, obj_t={type(obj_l)}')
                for obj, context in obj_l:
                    try:  # Error isolation between subobjects of one in_obj
                        ret_obj = self.process(obj, context)
                    except Exception as e:
                        log.exception(f'in OBJ process(obj=`{obj}`, context=`{context}`)')
                        obj.add_error(FebError(FebError.ET_SYS, self, {FebError.EC_EXCEPTION: e}))
                        ret_obj_l.append(obj)
                    else:
                        ret_obj_l.append(ret_obj)
                utt = self.pack_result(utt, ret_obj_l)
                # TODO: dump data:
                log.info(f'DATA DUMP utt=`{utt}`')
                var_dump(header = 'to_dump test', msg = utt.to_dump())
                pretty_json(utt.to_dump())

            except Exception as e:
                log.exception(f'in UTT process(utt=`{utt}`)')
                utt.add_error(FebError(FebError.ET_SYS, self, {FebError.EC_EXCEPTION: e}))
            if self.component_type() == self.FINAL_COMPONENT:
                res_batch.append(utt.return_text())
            else:
                res_batch.append(utt)
        return res_batch

    def test_and_prepare(self, utt):
        """
        Test input data and prepare data to process
        :param in_obj:
        :return: list (even if there is only one object to process!) of tuple(object, context)
            object - object for processing (must be instanceof FebObject)
            context - dictionary with context for processing
        """
        # [(FebObject(), {'empty_context': None})]
        # basic realization:
        # using utt as processing object
        return [(utt, {})]

    def process(self, obj, context):
        """
        Main processing function
        :param obj: obj to process
        :param context: dict with processing context
        :return: processed object
        """
        # basic realization:
        # doesn't change object
        return  obj

    def pack_result(self, utt, ret_obj_l):
        """
        Packing results of processing
        :param utt: current FebUtterance
        :param ret_obj_l: list of processed objects
        :return: utt with added values from ret_obj_l
        """
        # basic realization:
        # doesn't updated utt
        assert utt is ret_obj_l[0], 'Basic realization of pack_result() is incorrect!'
        return ret_obj_l[0]





