from typing import Tuple

import numpy as np


class PolicyPrediction:
    """
    Used to store policy model predictions and hidden values.
    """
    def __init__(self, probs, prediction, hidden_outs, cell_state):
        self.probs = probs
        self.prediction = prediction
        self.hidden_outs = hidden_outs
        self.cell_state = cell_state
        self.predicted_action_ix = np.argmax(probs)

    def get_network_state(self) -> Tuple:
        return self.cell_state, self.hidden_outs
