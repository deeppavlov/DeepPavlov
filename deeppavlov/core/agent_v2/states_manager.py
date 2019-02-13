from typing import Sequence, Hashable, List
from datetime import datetime

from deeppavlov.core.agent_v2.state_schema import User, Utterance, BotUtterance, DialogHistory, Dialog
from deeppavlov.core.agent_v2.connection import state_storage
from deeppavlov.core.agent_v2.agent import TG_START_UTT


class StatesManager:

    @staticmethod
    def get_user_states(user_telegram_ids: Sequence[Hashable], should_reset: Sequence[bool]):
        states = []
        for user_telegram_id, reset in zip(user_telegram_ids, should_reset):
            if reset:
                ...
            else:
                users = User.objects(user_telegram_id__in=user_telegram_ids)
                dialogs = Dialog.objects(users__in=users)
                states.append(dialogs)
        return states

    @staticmethod
    def create_new_state(user_telegram_id, bot_telegram_id, user_device_type=None, user_personality=None,
                         bot_device_type=None, bot_personality=None, channel_type='telegram'):
        user = User(user_telegram_id=user_telegram_id, device_type=user_device_type, personality=user_personality)
        bot = User(user_telegram_id=bot_telegram_id, bot_device_type=bot_device_type, bot_personality=bot_personality)
        utt = Utterance(text=TG_START_UTT, date=datetime.utcnow(), user=user)
        dh = DialogHistory([utt])
        d = Dialog(history=dh, users=[user, bot], channel_type=channel_type)
        user.save()
        bot.save()
        utt.save()
        ...






# TEST
# sm = StatesManager()
# sm.get_user_states(user_telegram_ids=["44d279ea-62ab-4c71-9adb-ed69143c12eb", "56f1d5b2-db1a-4128-993d-6cd1bc1b938f"])
