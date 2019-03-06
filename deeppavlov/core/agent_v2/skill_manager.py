from deeppavlov.core.agent_v2.config import MAX_WORKERS, SKILLS


class SkillManager:

    def __init__(self, response_selector, skill_caller, skill_selector=None):
        self.skill_selector = skill_selector
        self.response_selector = response_selector
        self.max_workers = MAX_WORKERS
        self.skill_caller = skill_caller
        self.skills = SKILLS

    def __call__(self, state):
        if self.skill_selector is not None:
            active_skill_names = self.skill_selector(state)
        else:
            active_skill_names = [s['name'] for s in self.skills]
        active_skill_urls = [s['url'] for s in self.skills if s['name'] in active_skill_names]
        skill_responses = self.skill_caller(payload=state, names=active_skill_names, urls=active_skill_urls)
        responses = self.response_selector(skill_responses, state)
        return responses
