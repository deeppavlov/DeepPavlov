
class Lettercaser(object):
    """It defines lettercases of tokens and can restore them.
    By default it detects only ['lower', 'upper', 'capitalize'] lettercases,
    but there is opportunity to expand that list with 'cases' argument
    Args:
        cases: dictionary that describes map,
               name of lettercase -> func that takes str and convert it in certain lettercase
        default_case: func: str->str that define transformation of string,
                      when lettercase was not be detected in 'put_in_case' func
    Attributes:
        cases: dictionary that describes map,
               name of lettercase -> func that takes str and convert it in certain lettercase
        default_case: func: str->str that define transformation of string,
                      when lettercase was not be detected in 'put_in_case' func
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

    def determine_lettercase(self, token):
        """It detemines case of token with 'cases' attribute
        Args:
            token: token lettercases of that have been detected
        """
        for case in self.cases:
            if token == self.cases[case](token):
                return case
        return None

    def put_in_lettercase(self, token: str, case: str):
        """It restore lettercases of tokens according to 'case' arg,
        if lettercase is not detected (case==None), 'default_case' func will be used
        Args:
            tokens: token that will be put in case
            case: name of lettercase
        Return:
            tokens in certain lettercases
            if lettercase was not detected then 'default_case'would be used
        """
        if case is None:
            return self.default_case(token)
        return self.cases[case](token)