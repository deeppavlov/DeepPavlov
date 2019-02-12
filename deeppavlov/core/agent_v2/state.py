from typing import List, Dict
import json
from pymongo import MongoClient

state_scheme = {'user_id': {'personality': None, 'device': None},
                'sess_id': None,
                'utt_id': None,
                'user_type': None,
                'annotations': {'ner': None,
                                'sentiment': None,
                                'coref': None,
                                'topic': None},
                'text': None,
                'confidence': None,
                'active_skill': None,
                'date': None,
                'location': None}


# class StateScheme:
#
#     def __init__(self):
#         self.user_id = {
#             "personality": None,
#             "device": None
#         }
#         self.sess_id = None
#         self.utt_id = None
#         self.user_type = "human"
#         self.annotations = {
#             "ner": None,
#             "sentiment": None,
#             "coref": None,
#             "topic": None
#         }
#         self.text = None
#         self.confidence = None
#         self.active_skill = None
#         self.date = None
#         self.location = None


class MongoConnector:
    def __init__(self, database, collection, host, port):
        self.database = database
        self.collection = collection
        self.host = host
        self.port = port

    def connect(self):
        connection = MongoClient(self.host, self.port)
        db = connection[self.database]
        collection = db[self.collection]
        return collection


class State:
    """Agent state includes preprocessor annotations, dialog state and skills internal states.

    State DB structure. Each row corresponds to a single dialog.

    | session_id          | user_id |           data       |
    +---------------------+---------+----------------------+
    | integer (unique)    | integer |    JSON string       |
    +---------------------+---------+----------------------+


    JSON-formatted data structure example:

    {
        "memory": {
            "session_id": 0,
            "user_id": 0,
            "user_data": ["location": "moscow", "device_name": "iphone"],
            "user_personality": "",
            "conversation_topic": "",
            "context":
                       [{"utt_id": 6, "utt_text": "привет", "type": "human",
                        "annotations": [{"ner": [{}], "coref": [{}], "sentiment": [{}]}], "confidence": 1.0},
                       {"utt_id": 7, "utt_text": "привет, я бот!", "type": "bot", "annotations": [{"ner": [{}],
                        "coref": [{}], "sentiment": [{}]}], "confidence": 0.952}]}]
            },
        "dialog_state": {"active_skill_history": ["chitchat", "chitchat", "odqa", "odqa", "chitchat"],
                        "active_skill": "chitchat", "active_skill_duration": 1, "dialog_duration": 6
            },
        "skill_state": [{"chitchat": [{}], "odqa": [{}], "go": [{}]}
    }

    """

    test_state = {
        "user_data": {
            "user_id"
            "personality",
            "device",
        },
        "sess_id"
        "utt_id"
        "user_type"
        "annotations": {
            "ner",
            "sentiment",
            "tokens",
            "coref",
            "topic"
        },
        "text"
        "confidence"
        "active_skill"
        "date"
        "location"
    }

    def __init__(self, connector):
        connection = connector.connect()

        # self.state = connection.get('state')
        self.state_scheme = state_scheme

    def update_annotations(self, annotations: Dict):
        self.state_scheme.annotations.update(**annotations)

    def update_text(self, text: str):
        self.state_scheme.text = text

    def get_session_state(self, sess_id):
        self.connection.get("user_id")

    def put_session_state(self):
        pass

    # def get_user_state(self, user_id):

    def dump_state(self):
        """
        Dump current state to DB.

        Returns: None

        """
        pass

    def get_all_dialogs(self):
        """
        Get all dialogs from the DB.

        Returns: JSON-formatted data from `data` db column

        """
        pass

    def get_dialogs_by_user(self, user_ids: List[int]):
        """
        Get all dialogs from the DB by specified users.

        Args:
            user_ids: list of ids

        Returns: JSON-formatted data from `data` db column

        """
        pass
