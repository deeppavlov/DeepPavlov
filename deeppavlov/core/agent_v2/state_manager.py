from typing import Sequence, Hashable, Any

from deeppavlov.core.agent_v2.state_schema import User, Human, Utterance, BotUtterance, Dialog
from deeppavlov.core.agent_v2.connection import state_storage
from deeppavlov.core.agent_v2.bot import BOT
from deeppavlov.core.agent_v2 import VERSION

TG_START_UTT = '/start'


class StateManager:

    def get_users(self, user_telegram_ids=Sequence[Hashable], user_device_types=Sequence[Any]):
        users = []
        for user_telegram_id, device_type in zip(user_telegram_ids, user_device_types):
            user_query = Human.objects(user_telegram_id__exact=user_telegram_id)
            if not user_query:
                user = self.create_new_human(user_telegram_id, device_type)
            else:
                user = user_query[0]
            users.append(user)
        return users

    def get_dialogs(self, users, utterances, locations, channel_types, should_reset):
        dialogs = []
        for user, utt, loc, channel_type, reset in zip(users, utterances,
                                                       locations, channel_types,
                                                       should_reset):
            if reset:
                new_dialog = self.create_new_dialog(users=[user, BOT],
                                                    utterances=[utt],
                                                    location=loc,
                                                    channel_type=channel_type)
                dialogs.append(new_dialog)
            else:
                d = Dialog.objects(users__in=[user])[0]
                d.utterances.append(utt)
                dialogs.append(d)
        return dialogs

    def get_utterances(self, texts, users, date_times, annotations=None):
        utterances = []
        if annotations is None:
            annotations = [None] * len(texts)
        for text, anno, user, date_time in zip(texts, annotations, users, date_times):
            utterances.append(self.create_new_utterance(text, user, date_time, anno))
        return utterances

    @staticmethod
    def get_state(dialogs):
        state = {'version': VERSION, 'dialogs': []}
        for d in dialogs:
            state['dialogs'].append(d.to_dict())
        return state

    @staticmethod
    def create_new_dialog(users, utterances, location=None, channel_type=None):
        dialog = Dialog(users=users,
                        utterances=utterances,
                        location=location or Dialog.location.default,
                        channel_type=channel_type
                        )
        dialog.save()
        return dialog

    @staticmethod
    def create_new_human(user_telegram_id, device_type, personality=None, profile=None):
        human = Human(user_telegram_id=user_telegram_id, device_type=device_type,
                      personality=personality, profile=profile or Human.profile.default)
        human.save()
        return human

    @staticmethod
    def create_new_utterance(text, user, date_time, annotations=None):
        if user.user_type == 'bot':
            raise RuntimeError(
                'Utterances of bots should be created with different method. See create_new_bot_utterance()')
        utt = Utterance(text=text, user=user, date_time=date_time,
                        annotations=annotations or Utterance.annotations.default)
        utt.save()
        return utt

    @staticmethod
    def create_new_bot_utterance(text, user, date_time, active_skill, confidence, annotations=None):
        utt = Utterance(text=text, user=user, date_time=date_time,
                        active_skill=active_skill, confidence=confidence,
                        annotations=annotations or BotUtterance.annotations.default)
        utt.save()
        return utt

    # TODO rewrite with using mongoengine.Document.update()
    @staticmethod
    def update_me_object(obj, **kwargs):
        for attr, value in kwargs.items():
            if not hasattr(obj, attr):
                raise AttributeError(f'{object.__class__.__name__} object doesn\'t have an attribute {attr}')
            setattr(obj, attr, value)
            obj.save()
