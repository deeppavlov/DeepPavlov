from datetime import datetime
import uuid
import json
from json import JSONEncoder
from bson import json_util

from mongoengine import Document, DynamicDocument, ReferenceField, ListField, StringField, DynamicField, \
    UUIDField, DateTimeField, FloatField, DictField

from mongoengine import connect

connect(host='localhost', port=27017)


class User(Document):
    # user_id = UUIDField(required=True) #
    user_telegram_id = UUIDField(required=True, unique=True)
    user_type = StringField(required=True, choices=['human', 'bot'], default='human')
    device_type = DynamicField()
    personality = DynamicField()

    def to_dict(self):
        return {'id': str(self.id),
                'user_telegram_id': str(self.user_telegram_id),
                'user_type': self.user_type,
                'device_type': self.device_type,
                'personality': self.personality}


class Utterance(Document):
    # utt_id = UUIDField(required=True) # managed by db
    channel_type = StringField(choices=['telegram', 'vkontakte', 'facebook'], default='telegram')
    text = StringField(required=True)
    annotations = DictField(default={'ner': {}, 'coref': {}, 'sentiment': {}})
    user = ReferenceField(User, required=True)

    meta = {'allow_inheritance': True}

    def to_dict(self):
        return {'id': str(self.id),
                'channel_type': self.channel_type,
                'text': self.text,
                'user_id': str(self.user.id),
                'annotations': self.annotations}


class DialogHistory(DynamicDocument):
    utterances = ListField(ReferenceField(Utterance), required=True)

    def to_dict(self):
        return {'id': str(self.id), 'utterances': [utt.to_dict() for utt in self.utterances]}


# class Annotations(Document):
#     annotations = DictField(required=True)
#     utterance = ReferenceField(Utterance, required=True)
#
#     def to_dict(self):
#         return {'id': self.id,
#                 'annotations': self.annotations,
#                 'utterance': self.utterance.to_dict()}


class BotUtterance(Utterance):
    active_skill = StringField()
    confidence = FloatField()

    def to_dict(self):
        return {
            'id': str(self.id),
            'active_skill': self.active_skill,
            'confidence': self.confidence,
            'channel_type': self.channel_type,
            'text': self.text,
            'user_id': str(self.user.id)
        }


class Dialog(DynamicDocument):
    # dialog_id = UUIDField(required=True) #managed by db
    location = DynamicField()
    date = DateTimeField(required=True)
    history = ReferenceField(DialogHistory, required=True)
    users = ListField(ReferenceField(User), required=True)

    def to_dict(self):
        return {
            'id': str(self.id),
            'location': self.location,
            'date': str(self.date),
            'history': self.history.to_dict(),
            'users': [u.to_dict() for u in self.users]
        }


class MongoEncoder(JSONEncoder):
    def default(self, obj):
        if issubclass(obj, Document):
            return obj.to_dict()
        return JSONEncoder.default(self, obj)

    # def __dict__(self):
    #     # return "%s %s %s %s" % (self.location, self.date, self.history, self.users)
    #     return {'location': self.location,
    #             'date': self.date,
    #             'history': self.history,
    #             'users': self.users}

    # def save(self, force_insert=False, validate=True, clean=True,
    #          write_concern=None, cascade=None, cascade_kwargs=None,
    #          _refs=None, save_condition=None, signal_kwargs=None, **kwargs):
    #     self.history.save()
    #     super().save()


# d = Dialog()
# string_of_user = "Hi"
# u = Utterance(text=string_of_user, annotations=[])
# u.save()
# Utterance.objects.filter(users=some_user)
# User.id

########################### Test case #######################################

default_anno = {"ner": [], "coref": [], "sentiment": []}
h_user = User(user_telegram_id=uuid.uuid4())
b_user = User(user_telegram_id=uuid.uuid4(), user_type='bot')

h_utt_1 = Utterance(text='Привет!', user=h_user, annotations=default_anno)
b_utt_1 = BotUtterance(text='Привет, я бот!', user=b_user, annotations=default_anno, active_skill='chitchat', confidence=0.85)
# h_anno_1 = Annotations(default_anno, utterance=h_utt_1)
# b_anno_1 = Annotations(default_anno, utterance=b_utt_1)

h_utt_2 = Utterance(channel_type='telegram', text='Как дела?', user=h_user, annotations=default_anno)
b_utt_2 = BotUtterance(channel_type='telegram', text='Хорошо, а у тебя как?', user=b_user, annotations=default_anno, active_skill='chitchat',
                       confidence=0.9333)
# h_anno_2 = Annotations(default_anno, utterance=h_utt_2)
# b_anno_2 = Annotations(default_anno, utterance=b_utt_2)

h_utt_3 = Utterance(text='И у меня нормально. Когда родился Петр Первый?', user=h_user, annotations=default_anno)
b_utt_3 = BotUtterance(text='в 1672 году', user=b_user, annotations=default_anno, active_skill='odqa', confidence=0.74)
# h_anno_3 = Annotations(default_anno, utterance=h_utt_3)
# b_anno_3 = Annotations(default_anno, utterance=b_utt_3)

dh = DialogHistory([h_utt_1, b_utt_1, h_utt_2, b_utt_2, h_utt_3, b_utt_3])
d = Dialog(date=datetime.utcnow(), history=dh, users=[h_user, b_user])

h_user.save()
b_user.save()

h_utt_1.save()
b_utt_1.save()

h_utt_2.save()
b_utt_2.save()

h_utt_3.save()
b_utt_3.save()

dh.save()
d.save()

for d in Dialog.objects:
    info = d.to_dict()
    total = {'version': 0.9, 'context': info}
    print(total)

