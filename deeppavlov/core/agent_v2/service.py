from typing import Dict


class Service:
    """Represent a basic service for the Agent.
       Agent can make calls to the following types of services:
       * annotators
       * skills
       * skill selectors
       * response selectors
       * postprocessors
    """
    def __init__(self, rest_caller):
        self.rest_caller = rest_caller

    def __call__(self, state: Dict):
        """
        Produce a response to a dialog state.

        Args:
            state: a dialog state

        Returns: a response

        """
        return self.rest_caller(state)
