from threading import Timer

from deeppavlov.core.common.chainer import Chainer


class BaseConversation:
    _config: dict
    _model: Chainer
    _conversation_lifetime: float
    _timer: Timer
    _infer_utterances: list

    def __init__(self, config: dict, model: Chainer, self_destruct_callback: callable):
        self._config = config
        self._model = model
        self._self_destruct_callback = self_destruct_callback
        self._conversation_lifetime = self._config['conversation_lifetime']
        self._infer_utterances = list()

    def _start_timer(self) -> None:
        """Initiates self-destruct timer."""
        self._timer = Timer(self._config['conversation_lifetime'], self._self_destruct_callback)
        self._timer.start()

    def _rearm_self_destruct(self) -> None:
        """Rearms self-destruct timer."""
        self._timer.cancel()
        self._start_timer()

    def _act(self, utterance: str) -> str:
        """Infers DeepPavlov model with raw user input extracted from request.

        Args:
            utterance: Raw user input extracted from request.

        Returns:
            response: DeepPavlov model response if  ``next_utter_msg`` from config with.

        """
        self._infer_utterances.append([utterance])
        if len(self._infer_utterances) == len(self._model.in_x):
            prediction = self._model(*self._infer_utterances)
            self._infer_utterances = list()
            if len(self._model.out_params) == 1:
                prediction = [prediction]
            prediction = '; '.join([str(output[0]) for output in prediction])
            response = prediction
        else:
            response = self._config['next_utter_msg'].format(self._model.in_x[len(self._infer_utterances)])
        return response
