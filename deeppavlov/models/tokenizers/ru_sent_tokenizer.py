from typing import Set, Tuple

from deeppavlov.core.models.component import Component
from deeppavlov.core.common.registry import register
from rusenttokenize import ru_sent_tokenize, SHORTENINGS, JOINING_SHORTENINGS, PAIRED_SHORTENINGS


@register("ru_sent_tokenizer")
class RuSentTokenizer(Component):
    """
    Rule-base sentence tokenizer for Russian language.
    https://github.com/deepmipt/ru_sentence_tokenizer
    """
    def __init__(self, shortenings: Set[str] = SHORTENINGS,
                       joining_shortenings: Set[str] = JOINING_SHORTENINGS,
                       paired_shortenings: Set[Tuple[str, str]] = PAIRED_SHORTENINGS):
        """
        Args:
            shortenings: list of known shortenings. Use default value if working on news or fiction texts
            joining_shortenings: list of shortenings after that sentence split is not possible (i.e. "ул").
                                 Use default value if working on news or fiction texts
            paired_shortenings: list of known paired shotenings (i.e. "т. е.").
                                Use default value if working on news or fiction texts

        """
        self.shortenings = shortenings
        self.joining_shortenings = joining_shortenings
        self.paired_shortenings = paired_shortenings

    def __call__(self, batch: [str]) -> [[str]]:
        return [ru_sent_tokenize(x, self.shortenings, self.joining_shortenings, self.paired_shortenings) for x in batch]
