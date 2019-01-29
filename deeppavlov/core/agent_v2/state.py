from typing import List


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

    def __init__(self, connector, db_conn: str = None, table: str = None, user_id_col: str = None,
                 sess_id_col: str = None, state_col: str = None):
        sess_state = connector(db_conn, table, user_id_col, sess_id_col, state_col)
        self.dialog_state = sess_state.get('dialog_state')
        self.memory = sess_state.get('memory')
        self.skill_state = sess_state.get('skill_state')

    def update_state(self, state):
        """
        Update dialog_state, memory and skill_state attributes.

        Args:
            state: current state

        Returns: None

        """
        pass

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
