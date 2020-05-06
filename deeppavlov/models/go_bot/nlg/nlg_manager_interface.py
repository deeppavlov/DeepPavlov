from abc import ABCMeta, abstractmethod


class NLGManagerInterface(metaclass=ABCMeta):

    @abstractmethod
    def get_action_id(self, action_text):
        """
        Looks up for an ID relevant to the passed action text in the list of known actions and their ids.
        :param action_text: the text for which an ID needs to be returned.
        :return: an ID corresponding to the passed action text
        """
        pass

    @abstractmethod
    def get_api_call_action_id(self):
        """
        :return: an ID corresponding to the api call action
        """
        pass

    @abstractmethod
    def decode_response(self, action_id, tracker_slotfilled_state):
        """
        Convert action template id and known slot values from tracker to response text.
        Replaces the unknown slot values with "dontcare" if the action is an API call.
        :param action_id: the id of action to generate text for.
        :param tracker_slotfilled_state: the slots and their known values. usually received from dialogue state tracker.

        :returns: the text generated for the passed action id and slot values.
        """
        pass

    @abstractmethod
    def num_of_known_actions(self):
        """
        :returns: the number of actions known to the NLG module
        """
        pass