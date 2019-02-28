from typing import Dict, List, Tuple


class ResponseSelector:
    def __init__(self, rest_caller=None):
        self.rest_caller = rest_caller

    def __call__(self, responses: Dict, state: Dict) -> Tuple[List[str], List[str], List[float]]:
        """
        Select a single response for each dialog in the state.

        Ex.:
        responses = [{'odqa': {'confidence': 3104.22607421875, 'text': 'опыты с магнитом'},
         'faq': {'confidence': 0.28, 'text': 'Рекомендация преподавателя центра "Физтех-Потенциал"
          не даёт льготы при поступлении.'}},
          {'odqa': {'confidence': 23.78227424621582, 'text': 'британский империализм'},
          'faq': {'confidence': nan, 'text': 'Формы заявления и согласия будут размещены по ссылке.'}},
           {'odqa': {'confidence': 2.298262119293213, 'text': '33 символ в ASCII'},
            'faq': {'confidence': nan, 'text': 'Формы заявления и согласия будут размещены по ссылке.'}}]

        Args:
            responses:
            state:

        Returns: a list of skill names, a list of response utterances, a list of condfidences

        """
        raise NotImplementedError


class ConfidenceResponseSelector(ResponseSelector):

    def __init__(self):
        super().__init__()

    def __call__(self, responses: Dict, state: Dict) -> Tuple[List[str], List[str], List[float]]:
        skill_names = []
        utterances = []
        confidences = []
        for r in responses:
            sr = sorted(r.items(), key=lambda x: x[1]['confidence'], reverse=True)[0]
            skill_names.append(sr[0])
            utterances.append(sr[1]['text'])
            confidences.append(sr[1]['confidence'])
        return skill_names, utterances, confidences
