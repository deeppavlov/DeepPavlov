from mongoengine import DynamicDocument, ReferenceField, ListField, StringField, DynamicField, \
    DateTimeField, FloatField, DictField


class User(DynamicDocument):
    personality = DynamicField()

    meta = {'allow_inheritance': True}

    def to_dict(self):
        raise NotImplementedError


class Bot(User):
    def to_dict(self):
        return {'id': str(self.id),
                'user_type': 'bot',
                'personality': self.personality}


class Human(User):
    user_telegram_id = StringField(required=True, unique=True, sparse=True)
    device_type = DynamicField()
    profile = DictField(required=True, default={
        "name": None,
        "gender": None,
        "birthdate": None,
        "location": None,
        "home_coordinates": None,
        "work_coordinates": None,
        "occupation": None,
        "income_per_year": None
    })

    def to_dict(self):
        return {'id': str(self.id),
                'user_telegram_id': str(self.user_telegram_id),
                'user_type': 'human',
                'device_type': self.device_type,
                'personality': self.personality,
                'profile': self.profile}


class Utterance(DynamicDocument):
    text = StringField(required=True)
    annotations = DictField(default={'ner': {}, 'coref': {}, 'sentiment': {}})
    user = ReferenceField(User, required=True)
    date_time = DateTimeField(required=True)

    meta = {'allow_inheritance': True}

    def to_dict(self):
        return {'id': str(self.id),
                'text': self.text,
                'user_id': str(self.user.id),
                'annotations': self.annotations,
                'date_time': str(self.date_time)}


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
            'date_time': str(self.date_time)
        }


class Dialog(DynamicDocument):
    location = DynamicField()
    utterances = ListField(ReferenceField(Utterance), required=True, default=[])
    user = ReferenceField(Human, required=True)
    bot = ReferenceField(Bot, required=True)
    channel_type = StringField(choices=['telegram', 'vk', 'facebook'], default='telegram')

    def to_dict(self):
        return {
            'id': str(self.id),
            'location': self.location,
            'utterances': [utt.to_dict() for utt in self.utterances],
            'user': self.user.to_dict(),
            'bot': self.bot.to_dict(),
            'channel_type': self.channel_type
        }
