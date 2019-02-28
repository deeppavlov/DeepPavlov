from deeppavlov.core.agent_v2.config import MAX_WORKERS, SKILLS


class SkillManager:

    def __init__(self, response_selector, rest_caller, skills_selector=None):
        self.skills_selector = skills_selector
        self.response_selector = response_selector
        self.max_workers = MAX_WORKERS
        self.rest_caller = rest_caller
        self.skills = SKILLS

    def __call__(self, state):
        if self.skills_selector is not None:
            active_skills = self.skills_selector.select_skills(self.skills, state)
        else:
            active_skills = self.skills
        active_skill_names = [s['name'] for s in active_skills]
        active_skill_urls = [s['url'] for s in active_skills]
        skill_responses = self.rest_caller(active_skill_names, active_skill_urls, state)
        responses = self.response_selector(skill_responses, state)
        return responses
