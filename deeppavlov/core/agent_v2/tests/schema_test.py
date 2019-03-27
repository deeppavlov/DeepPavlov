from datetime import datetime
import uuid

from deeppavlov.core.agent_v2.state_schema import Human, Bot, Utterance, BotUtterance, Dialog
from deeppavlov.core.agent_v2.connection import state_storage
from deeppavlov.core.agent_v2.bot import BOT

########################### Test case #######################################

# User.drop_collection()
Human.drop_collection()

Dialog.objects.delete()
Utterance.objects.delete()
BotUtterance.objects.delete()
# User.objects.delete()
Human.objects.delete()


h_user = Human(user_telegram_id=str(uuid.uuid4()))

h_utt_1 = Utterance(text='Привет!', user=h_user, date_time=datetime.utcnow())
b_utt_1 = BotUtterance(text='Привет, я бот!', user=BOT, active_skill='chitchat',
                       confidence=0.85, date_time=datetime.utcnow())

h_utt_2 = Utterance(text='Как дела?', user=h_user, date_time=datetime.utcnow())
b_utt_2 = BotUtterance(text='Хорошо, а у тебя как?', user=BOT,
                       active_skill='chitchat',
                       confidence=0.9333, date_time=datetime.utcnow())

h_utt_3 = Utterance(text='И у меня нормально. Когда родился Петр Первый?', user=h_user, date_time=datetime.utcnow())
b_utt_3 = BotUtterance(text='в 1672 году', user=BOT, active_skill='odqa', confidence=0.74,
                       date_time=datetime.utcnow())
print(b_utt_3.to_dict())


# for d in Dialog.objects:
#    print(d.to_dict())

state = {'version': '0.10.1', 'dialogs': []}
for d in Dialog.objects:
    state['dialogs'].append(d.to_dict())

print(state)


