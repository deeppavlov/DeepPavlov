from mongoengine import Document, DynamicDocument, ReferenceField, ListField, StringField, DynamicField, \
    UUIDField, DateTimeField, FloatField, DictField


class User(Document):
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
    text = StringField(required=True)
    annotations = DictField(default={'ner': {}, 'coref': {}, 'sentiment': {}})
    user = ReferenceField(User, required=True)
    date = DateTimeField(required=True)

    meta = {'allow_inheritance': True}

    def to_dict(self):
        return {'id': str(self.id),
                'text': self.text,
                'user_id': str(self.user.id),
                'annotations': self.annotations,
                'date': str(self.date)}


class DialogHistory(DynamicDocument):
    utterances = ListField(ReferenceField(Utterance), required=True)

    def to_dict(self):
        return {'utterances': [utt.to_dict() for utt in self.utterances]}


class BotUtterance(Utterance):
    active_skill = StringField()
    confidence = FloatField()

    def to_dict(self):
        return {
            'id': str(self.id),
            'active_skill': self.active_skill,
            'confidence': self.confidence,
            'text': self.text,
            'user_id': str(self.user.id),
            'annotations': self.annotations,
            'date': str(self.date)
        }


class Dialog(DynamicDocument):
    location = DynamicField()
    history = ReferenceField(DialogHistory, required=True)
    users = ListField(ReferenceField(User), required=True)
    channel_type = StringField(choices=['telegram', 'vkontakte', 'facebook'], default='telegram')

    def to_dict(self):
        return {
            'id': str(self.id),
            'location': self.location,
            'history': self.history.to_dict(),
            'user': [u.to_dict() for u in self.users if u.user_type == 'human'][0],
            'bot': [u.to_dict() for u in self.users if u.user_type == 'bot'][0],
            'channel_type': self.channel_type
        }
