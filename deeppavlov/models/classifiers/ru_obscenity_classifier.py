from typing import List, Union
from pathlib import Path
import json
import pymorphy2
import re
from logging import getLogger

from deeppavlov.core.common.registry import register
from deeppavlov.core.models.estimator import Component
from deeppavlov.core.commands.utils import expand_path

log = getLogger(__name__)


@register("ru_obscenity_classifier")
class RuObscenityClassifier(Component):
    """Rule-Based model that decides whether the sentence is obscene or not,
    for russian language

    Args:
        data_path: a directory where required files are storing
    Attributes:
        obscenity_words: list of russian obscenity words
        obscenity_words_extended: list of russian obscenity words
        obscenity_words_exception: list of words on that model makes mistake that they are obscene
        regexp: reg exp that finds various obscene words
        regexp2: reg exp that finds various obscene words
        morph: pymorphy2.MorphAnalyzer object
        word_pattern: reg exp that finds words in text
    """

    def _get_patterns(self):
        PATTERN_1 = r''.join((
            r'\w{0,5}[хx]([хx\s\!@#\$%\^&*+-\|\/]{0,6})',
            r'[уy]([уy\s\!@#\$%\^&*+-\|\/]{0,6})[ёiлeеюийя]\w{0,7}|\w{0,6}[пp]',
            r'([пp\s\!@#\$%\^&*+-\|\/]{0,6})[iие]([iие\s\!@#\$%\^&*+-\|\/]{0,6})',
            r'[3зс]([3зс\s\!@#\$%\^&*+-\|\/]{0,6})[дd]\w{0,10}|[сcs][уy]',
            r'([уy\!@#\$%\^&*+-\|\/]{0,6})[4чkк]\w{1,3}|\w{0,4}[bб]',
            r'([bб\s\!@#\$%\^&*+-\|\/]{0,6})[lл]([lл\s\!@#\$%\^&*+-\|\/]{0,6})',
            r'[yя]\w{0,10}|\w{0,8}[её][bб][лске@eыиаa][наи@йвл]\w{0,8}|\w{0,4}[еe]',
            r'([еe\s\!@#\$%\^&*+-\|\/]{0,6})[бb]([бb\s\!@#\$%\^&*+-\|\/]{0,6})',
            r'[uу]([uу\s\!@#\$%\^&*+-\|\/]{0,6})[н4ч]\w{0,4}|\w{0,4}[еeё]',
            r'([еeё\s\!@#\$%\^&*+-\|\/]{0,6})[бb]([бb\s\!@#\$%\^&*+-\|\/]{0,6})',
            r'[нn]([нn\s\!@#\$%\^&*+-\|\/]{0,6})[уy]\w{0,4}|\w{0,4}[еe]',
            r'([еe\s\!@#\$%\^&*+-\|\/]{0,6})[бb]([бb\s\!@#\$%\^&*+-\|\/]{0,6})',
            r'[оoаa@]([оoаa@\s\!@#\$%\^&*+-\|\/]{0,6})[тnнt]\w{0,4}|\w{0,10}[ё]',
            r'([ё\!@#\$%\^&*+-\|\/]{0,6})[б]\w{0,6}|\w{0,4}[pп]',
            r'([pп\s\!@#\$%\^&*+-\|\/]{0,6})[иeеi]([иeеi\s\!@#\$%\^&*+-\|\/]{0,6})',
            r'[дd]([дd\s\!@#\$%\^&*+-\|\/]{0,6})[oоаa@еeиi]',
            r'([oоаa@еeиi\s\!@#\$%\^&*+-\|\/]{0,6})[рr]\w{0,12}',
        ))

        PATTERN_2 = r'|'.join((
            r"(\b[сs]{1}[сsц]{0,1}[uуy](?:[ч4]{0,1}[иаakк][^ц])\w*\b)",
            r"(\b(?!пло|стра|[тл]и)(\w(?!(у|пло)))*[хx][уy](й|йа|[еeё]|и|я|ли|ю)(?!га)\w*\b)",
            r"(\b(п[oо]|[нз][аa])*[хx][eе][рp]\w*\b)",
            r"(\b[мm][уy][дd]([аa][кk]|[oо]|и)\w*\b)",
            r"(\b\w*д[рp](?:[oо][ч4]|[аa][ч4])(?!л)\w*\b)",
            r"(\b(?!(?:кило)?[тм]ет)(?!смо)[а-яa-z]*(?<!с)т[рp][аa][хx]\w*\b)",
            r"(\b[кk][аaoо][з3z]+[eе]?ё?л\w*\b)",
            r"(\b(?!со)\w*п[еeё]р[нд](и|иc|ы|у|н|е|ы)\w*\b)",
            r"(\b\w*[бп][ссз]д\w+\b)",
            r"(\b([нnп][аa]?[оo]?[xх])\b)",
            r"(\b([аa]?[оo]?[нnпбз][аa]?[оo]?)?([cс][pр][аa][^зжбсвм])\w*\b)",
            r"(\b\w*([оo]т|вы|[рp]и|[оo]|и|[уy]){0,1}([пnрp][iиеeё]{0,1}[3zзсcs][дd])\w*\b)",
            r"(\b(вы)?у?[еeё]?би?ля[дт]?[юоo]?\w*\b)",
            r"(\b(?!вело|ски|эн)\w*[пpp][eеиi][дd][oaоаеeирp](?![цянгюсмйчв])[рp]?(?![лт])\w*\b)",
            r"(\b(?!в?[ст]{1,2}еб)(?:(?:в?[сcз3о][тяaа]?[ьъ]?|вы|п[рp][иоo]|[уy]|р[aа][з3z][ьъ]?|к[оo]н[оo])?[её]б[а-яa-z]*)|(?:[а-яa-z]*[^хлрдв][еeё]б)\b)",
            r"(\b[з3z][аaоo]л[уy]п[аaeеин]\w*\b)",
        ))

        return PATTERN_1, PATTERN_2

    def __init__(self, data_path: Union[Path, str], *args, **kwargs):
        log.info(f"Initializing `{self.__class__.__name__}`")
        data_path = Path(expand_path(data_path))
        required_files = ['obscenity_words.json',
                          'obscenity_words_exception.json',
                          'obscenity_words_extended.json']
        for file in required_files:
            if not (data_path/file).exists():
                raise FileNotFoundError(errno.ENOENT,
                                        os.strerror(errno.ENOENT),
                                        str(data_path/file))

        self.obscenity_words = set(json.load(
            open(data_path/'obscenity_words.json', encoding="utf-8")))
        self.obscenity_words_extended = set(json.load(
            open(data_path/'obscenity_words_extended.json', encoding="utf-8")))
        self.obscenity_words_exception = set(json.load(
            open(data_path/'obscenity_words_exception.json', encoding="utf-8")))
        self.obscenity_words.update(self.obscenity_words_extended)

        PATTERN_1, PATTERN_2 = self._get_patterns()
        self.regexp = re.compile(PATTERN_1, re.U | re.I)
        self.regexp2 = re.compile(PATTERN_2, re.U | re.I)
        self.morph = pymorphy2.MorphAnalyzer()
        self.word_pattern = re.compile(r'[А-яЁё]+')

    def _check_obscenity(self, text):
        for word in self.word_pattern.findall(text):
            if len(word) < 3:
                continue
            word = word.lower()
            word.replace('ё', 'е')
            normal_word = self.morph.parse(word)[0].normal_form
            if normal_word in self.obscenity_words_exception\
                    or word in self.obscenity_words_exception:
                continue
            if normal_word in self.obscenity_words\
                    or word in self.obscenity_words\
                    or bool(self.regexp.findall(normal_word))\
                    or bool(self.regexp.findall(word))\
                    or bool(self.regexp2.findall(normal_word))\
                    or bool(self.regexp2.findall(word)):
                return 'obscene'
        return 'not_obscene'

    def __call__(self, texts: List[str]) -> List[str]:
        """it decides whether text is obscene or not

        Args:
            texts: list of texts, for which it needs to decide they are obscene or not

        Returns:
            list of strings: 'obscene' or 'not_obscene'
        """
        decisions = list(map(self._check_obscenity, texts))
        return decisions