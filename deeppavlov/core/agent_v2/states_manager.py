from typing import Sequence, Hashable, List, Tuple, Any
from datetime import datetime

from deeppavlov.core.agent_v2.state_schema import User, Human, Utterance, BotUtterance, Dialog
from deeppavlov.core.agent_v2.connection import state_storage
from deeppavlov.core.agent_v2.bot import BOT

TG_START_UTT = '\\start'


class StatesManager:

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

    def get_states(self, users, utterances, locations, channel_types, should_reset):
        states = []
        for user, utt, loc, channel_type, reset in zip(users, utterances,
                                                       locations, channel_types,
                                                       should_reset):
            if reset:
                new_dialog = self.create_new_dialog(users=[user, BOT],
                                                    utterances=[utt],
                                                    location=loc,
                                                    channel_type=channel_type)
                states.append(new_dialog)
            else:
                dialog_query = Dialog.objects(users__in=[user])
                states.append(dialog_query[0])
        return states

    def get_utterances(self, texts, annotations, users, date_times):
        utterances = []
        for text, anno, user, date_time in zip(texts, annotations, users, date_times):
            utterances.append(self.create_new_utterance(text, user, date_time, anno))
        return utterances

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
    def create_new_human(user_telegram_id, device_type, personality=None):
        human = Human(user_telegram_id=user_telegram_id, device_type=device_type,
                      personality=personality)
        human.save()
        return human

    @staticmethod
    def create_new_utterance(text, user, date_time, annotations=None):
        utt = Utterance(text=text, user=user, date_time=date_time,
                        annotations=annotations or Utterance.annotations.default)
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

    def save_state(self):
        """Save dialogs and all related objects to DB.
        Returns:

        """
        ...
