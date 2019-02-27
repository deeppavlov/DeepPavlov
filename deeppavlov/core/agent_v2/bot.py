from deeppavlov.core.agent_v2.state_schema import User
from deeppavlov.core.agent_v2.connection import connect

try:
    BOT = User.objects(id__exact='5c68347a0110b32ee73b2d97')[0]
except IndexError:
    BOT = User(id='5c68347a0110b32ee73b2d97', user_type='bot')
    BOT.save()
