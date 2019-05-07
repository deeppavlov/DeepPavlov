
class Lettercaser:
    """It defines lettercases of tokens and can restore them.
    By default it detects only ['lower', 'upper', 'capitalize'] lettercases,
    but there is opportunity to expand that list with 'cases' argument
    Args:
        cases: dictionary that describes map,
               name of lettercase -> func that takes str and convert it in certain lettercase
        default_case: func: str->str that define transformation of string when lettercase was not be detected
    Attributes:
        cases: dictionary that describes map,
               name of lettercase -> func that takes str and convert it in certain lettercase
        default_case: func: str->str that define transformation of string when lettercase was not be detected
    """

    def __init__(self, cases: dict = None, default_case = None):
        if default_case is None:
            self.default_case = lambda x: x.lower()
        else:
            self.default_case = default_case
        if cases is None:
            self.cases = {
                "lower": lambda x: x.lower(),
                "capitalize": lambda x: x.capitalize(),
                "upper": lambda x: x.upper()
            }
        else:
            self.cases = cases

    def get_case(self, token):
        """It detects case of token with 'cases' attribute
        Args:
            token: token lettercases of that will be detected
        """
        for case in self.cases:
            if token == self.cases[case](token):
                return case
        return None

    def put_in_case(self, token: str, case: str):
        """It restore lettercases of tokens according to 'case' arg,
        if lettercase was not detected (case==None), 'default_case' func would be used
        Args:
            tokens: token that will be put in case
            case: name of lettercase
        Return:
            tokens in certain lettercase
            if lettercase was not detected then 'default_case'would be used
        """
        if case is None:
            return self.default_case(token)
        return self.cases[case](token)