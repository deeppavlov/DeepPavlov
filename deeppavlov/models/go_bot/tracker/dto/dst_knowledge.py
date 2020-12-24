from deeppavlov.models.go_bot.tracker.dto.tracker_knowledge_interface import TrackerKnowledgeInterface


# todo naming
class DSTKnowledge(TrackerKnowledgeInterface):
    def __init__(self, tracker_prev_action, state_features, context_features, api_call_id, n_actions, action_mask):
        self.tracker_prev_action = tracker_prev_action
        self.state_features = state_features
        self.context_features = context_features
        self.api_call_id = api_call_id
        self.n_actions = n_actions
        self.action_mask = action_mask
