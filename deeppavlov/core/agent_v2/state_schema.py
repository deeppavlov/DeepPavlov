from mongoengine import DynamicDocument, ReferenceField, ListField, StringField, DynamicField, \
    DateTimeField, FloatField, DictField, CachedReferenceField


class User(DynamicDocument):
    persona = ListField(default=[])

    meta = {'allow_inheritance': True}

    def to_dict(self):
        raise NotImplementedError


class Bot(User):
    persona = ListField(default=['Мне нравится общаться с людьми.',
                                 'Пару лет назад я окончила вуз с отличием.',
                                 'Я работаю в банке.',
                                 'В свободное время помогаю пожилым людям в благотворительном фонде',
                                 'Люблю путешествовать'])

    def to_dict(self):
        return {'id': str(self.id),
                'user_type': 'bot',
                'persona': self.persona,
                }


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
                'persona': self.persona,
                'profile': self.profile}


class Utterance(DynamicDocument):
    text = StringField(required=True)
    annotations = DictField(default={'ner': {}, 'coref': {}, 'sentiment': {}, 'obscenity': {}})
    user = ReferenceField(User, required=True)
    date_time = DateTimeField(required=True)

    meta = {'allow_inheritance': True}

    def to_dict(self):
        raise NotImplementedError


class HumanUtterance(Utterance):
    selected_skills = DynamicField(default=[])

    def to_dict(self):
        return {'id': str(self.id),
                'text': self.text,
                'user_id': str(self.user.id),
                'annotations': self.annotations,
                'date_time': str(self.date_time),
                'selected_skills': self.selected_skills}


class BotUtterance(Utterance):
    active_skill = StringField()
    user = ReferenceField(Bot, required=True)
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
    utterances = ListField(default=[])
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
