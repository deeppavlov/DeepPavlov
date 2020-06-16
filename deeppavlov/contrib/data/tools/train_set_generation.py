import json
import deeppavlov.models.go_bot.nlg.templates.templates as templ
from deeppavlov.models.go_bot.nlg.templates.templates import DefaultTemplate
from logging import getLogger
from deeppavlov.models.slotfill.slotfill_raw import SlotFillingComponent
from deeppavlov.core.data.sqlite_database import Sqlite3Database
from deeppavlov.models.go_bot.tracker.dialogue_state_tracker import DialogueStateTracker
from deeppavlov.models.go_bot.nlu.dto.nlu_response import NLUResponse
from typing import List
import re
import itertools
log = getLogger(__name__)


class TrainSetGeneration():
    """
    Generates train dataset in the DSTC2 format via a command line.

    Args:
        tempalate_path = path to the go_bot template (in case of restaurant gobot, it is 'dstc2-templates.txt')
        slot_path = path to the slot values path (in case of restaurant gobot, it is 'dstc_slot_vals.json')
        save_path = this path is where you want to save the end result
        db_path = path to the populated database from which results are retrieved
        db_primary_key = the primary key of the database
    """

    def __init__(self,
                 template_path: str,
                 slot_path: str,
                 save_path: str,
                 db_path: str,
                 db_primary_key: List[str] = ['name']):
        self.templates = templ.Templates(DefaultTemplate()).load(template_path)
        self.slotfiller = SlotFillingComponent(load_path=slot_path, save_path=slot_path)
        self.save_path = save_path
        self.database = Sqlite3Database(db_path, db_primary_key)
        self.ds_tracker = DialogueStateTracker(api_call_id=0, slot_names=list(self.slotfiller._slot_vals.keys()),
                                               n_actions=len(self.templates.actions),
                                               hidden_size=128,
                                               database=self.database)
        self.slots = list(set(list(itertools.chain.from_iterable(map(lambda x: re.findall(r"#(\w+)", x.text), self.templates.templates)))
                              + list(self.slotfiller._slot_vals.keys())))
        self.utters = []
        self.dialogs = []
        self.slots_history = {}


    def get_id_input(self,
                     prompt: str,
                     valid_vals: List[int]) -> int:
        # for neat output
        print('\n' + '*' * 10)
        idx = -1
        while idx == -1:
            try:
                idx = int(input('[INPUT] ' + prompt))
                if not idx in valid_vals:
                    print('[INFO] please input a valid integer in: ', valid_vals)
                    idx = -1
            except ValueError:
                print('[INFO] please enter integer value')
        return idx

    def save_dialogs(self) -> None:
        from pathlib import Path
        with open(Path(self.save_path), 'w', encoding='utf8') as f:
            print('[INFO] saving the dialogs and exiting...')
            json.dump(self.dialogs, f)

    def add_and_reset_utters(self) -> None:
        if self.utters:
            self.dialogs.append(self.utters)
            self.utters = []
            self.slots_history = {}
            self.ds_tracker.reset_state()
        else:
            self.dialogs.append([])



    def get_user_input(self) -> None:
        text = input('[INFO] write a user sentence: ')
        has_slot = self.get_id_input(prompt = 'type 1 if your sentence has a slot, else 0: ',
                                     valid_vals = [0, 1])
        slots = []
        if has_slot:

            while has_slot:
                for i, key in enumerate(self.slots):
                    print(i, key)
                idx = self.get_id_input(prompt = 'type slot category number from the list: ',
                                        valid_vals = list(range(len(self.slots))))
                slot_category = self.slots[idx]
                if slot_category in self.slotfiller._slot_vals:
                    id2key = {}
                    for i, key in enumerate(self.slotfiller._slot_vals[slot_category]):
                        print(i, key)
                        id2key[i]=key
                    idx = self.get_id_input(prompt = 'type slot subcategory number from the list: ',
                                            valid_vals = list(range(len(id2key))))
                    sub_category = id2key[idx]
                else:
                    sub_category = ''

                slots.append([slot_category, sub_category])
                has_slot = self.get_id_input(prompt = 'type 1 if you want to add more slots, else 0: ',
                                             valid_vals = [0, 1])

        user_input = {'speaker': 1,
                      'text': text,
                      'slots': slots}

        print(user_input)
        self.update_slots_history(slots)
        self.utters.append(user_input)

    def update_slots_history(self, slots: List[List[str]]) -> None:
        for slot, val in slots:
            self.slots_history[slot] = val

    def start_generation(self) -> None:

        while True:
            turn = self.get_id_input(prompt = 'choose turn (1 for user, 2 for bot, 3 to start a new dialog or 10 for saving and exit): ',
                                     valid_vals = [1, 2, 3, 10])
            print('\n' + '*' * 10)
            if turn == 1:
                self.get_user_input()
            elif turn == 2:
                self.get_bot_output()
            elif turn == 3:
                self.add_and_reset_utters()
            elif turn == 10:
                self.add_and_reset_utters()
                self.save_dialogs()
                return


    def get_bot_output(self) -> None:

        print('[INFO] current slot vals are: ', self.slots_history)
        for i, act in enumerate(self.templates.actions):
            print(i, act)
        id = int(input('type template number from the list: '))
        #slots in the template chosen
        template_slots = re.findall(r"#(\w+)", self.templates.templates[id].text)
        slots = [[slot, self.slots_history[slot]] for slot in template_slots if slot in self.slots_history and slot in self.slotfiller._slot_vals]
        # slots that are missing int the current slots history
        missing_slots = [st for st in template_slots if st not in slots]
        # get missing slots from the db
        if missing_slots and self.ds_tracker.db_result:
            for slot in missing_slots:
                slots.append([slot, self.ds_tracker.db_result[slot]])
        text = self.templates.templates[id].generate_text(slots).strip()
        print('[INFO] generated response is: ', text)
        # make db call if 'api_call'
        if 'api_call' in self.templates.templates[id].text:
            nlu_response = NLUResponse(slots,None, None)
            self.ds_tracker.update_state(nlu_response)
            self.ds_tracker.make_api_call()
            print('[INFO] the result of the db call is: ', self.ds_tracker.db_result)

            bot_output = {'speaker': 2,
                  'text': text,
                  'db_result': json.dumps(self.ds_tracker.db_result),
                  'slots': slots,
                  'act': self.templates.actions[id]}
        else:
            bot_output = {'speaker': 2,
                  'text': text,
                  'slots': slots,
                  'act': self.templates.actions[id]}
        self.utters.append(bot_output)




