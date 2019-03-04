class Preprocessor:
    def __init__(self, rest_caller=None):
        self.rest_caller = rest_caller

    def __call__(self, utterances, should_reset):
        raise NotImplementedError


class IndependentPreprocessor(Preprocessor):
    """
    Run all annotators independently from each other.
    """

    def __init__(self, rest_caller):
        super().__init__(rest_caller)

    def __call__(self, dialogs: dict, should_reset):
        annotations = iter(self.rest_caller(dialogs))
        res = []
        for reset in should_reset:
            res.append(None if reset else next(annotations))
        return res
