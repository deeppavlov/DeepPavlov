from mongoengine import Document, DynamicDocument, ReferenceField, ListField, StringField, DynamicField, \
    UUIDField, DateTimeField, FloatField, DictField


class User(Document):
    user_type = StringField(required=True, choices=['human', 'bot'], default='human')
    personality = DynamicField()

    meta = {'allow_inheritance': True}

    def to_dict(self):
        return {'id': str(self.id),
                'user_type': self.user_type,
                'personality': self.personality}

    @classmethod
    def get_or_create(cls, *args, **kwargs):
        """
        gets or creates an object from init specification

        Args:
            *args:
            kwargs:

        Returns: tuple (instance:User, is_created:bool)
            or raises Exception if Multiple instances are found
        """
        results = cls.objects(*args, **kwargs)
        if results:
            if len(results)>1:
                # raise Exception
                raise Exception("Multiple instances found for %s: (%s, %s)!" % (cls.__name__, args, kwargs))
            elif len(results)==1:
                # ok
                return results[0], False
        else:
            # need to create an instance:
            instance = cls(*args, **kwargs)
            instance.save()
            return instance, True


class Human(User):
    user_telegram_id = UUIDField(required=True, unique=True, sparse=True)
    device_type = DynamicField()

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
    users = ListField(ReferenceField(User), required=True)
    channel_type = StringField(choices=['telegram', 'vkontakte', 'facebook'], default='telegram')

    def to_dict(self):
        return {
            'id': str(self.id),
            'location': self.location,
            'utterances': [utt.to_dict() for utt in self.utterances],
            'user': [u.to_dict() for u in self.users if u.user_type == 'human'][0],
            'bot': [u.to_dict() for u in self.users if u.user_type == 'bot'][0],
            'channel_type': self.channel_type
        }


class SkillResponse(DynamicDocument):
    """
    Data Model for Skills responses. It specifies behavior hypothesis from particular skill
    after processing new Utternace
    """
    text = StringField()
    confidence = FloatField()

    def to_dict(self):
        return {
            'id': str(self.id),
            'text': self.text,
            'confidence': self.confidence
        }
