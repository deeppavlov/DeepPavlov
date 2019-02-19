from deeppavlov.core.agent_v2.rest_caller import RestCaller
from deeppavlov.core.agent_v2.config import AGENT_CONFIG


class SkillManager:

    def __init__(self, skills_selector, response_selector, rest_caller: RestCaller):
        self.skills_selector = skills_selector
        self.response_selector = response_selector
        self.rest_caller = RestCaller(self.max_workers)
        self.skills = AGENT_CONFIG['skills']
        self.max_workers = AGENT_CONFIG['max_workers']

    def __call__(self, state):
        active_skills = self.skills_selector.select_skills(self.skills, state)
        active_skill_names = [s['name'] for s in active_skills]
        active_skill_urls = [s['url'] for s in active_skills]
        skill_responses = self.rest_caller(active_skill_names, active_skill_urls, state)
        response = self.response_selector.select_response(skill_responses, state)
        return response
