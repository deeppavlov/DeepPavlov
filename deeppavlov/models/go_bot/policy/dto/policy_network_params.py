from logging import getLogger

log = getLogger(__name__)


class PolicyNetworkParams:
    """
    The class to deal with the overcomplicated structure of the GO-bot configs.
    It is initialized from the config-as-is and performs all the conflicting parameters resolution internally.
    """
    # todo remove the complex config logic
    UNSUPPORTED = ["obs_size"]
    DEPRECATED = ["end_learning_rate", "decay_steps", "decay_power"]

    def __init__(self,
                 hidden_size,
                 dropout_rate,
                 l2_reg_coef,
                 dense_size,
                 attention_mechanism,
                 network_parameters):
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.l2_reg_coef = l2_reg_coef
        self.dense_size = dense_size
        self.attention_mechanism = attention_mechanism
        self.network_parameters = network_parameters or {}

        self.log_deprecated_params(self.network_parameters.keys())

    def get_hidden_size(self):
        return self.network_parameters.get("hidden_size", self.hidden_size)

    def get_action_size(self):
        return self.network_parameters.get("action_size")

    def get_dropout_rate(self):
        return self.network_parameters.get("dropout_rate", self.dropout_rate)

    def get_l2_reg_coef(self):
        return self.network_parameters.get("l2_reg_coef", self.l2_reg_coef)

    def get_dense_size(self):
        return self.network_parameters.get("dense_size", self.dense_size) or self.hidden_size  # todo :(

    def get_learning_rate(self):
        return self.network_parameters.get("learning_rate", None)

    def get_attn_params(self):
        return self.network_parameters.get('attention_mechanism', self.attention_mechanism)

    def log_deprecated_params(self, network_parameters):
        if any(p in network_parameters for p in self.DEPRECATED):
            log.warning(f"parameters {self.DEPRECATED} are deprecated,"
                        f" for learning rate schedule documentation see"
                        f" deeppavlov.core.models.lr_scheduled_tf_model"
                        f" or read a github tutorial on super convergence.")
