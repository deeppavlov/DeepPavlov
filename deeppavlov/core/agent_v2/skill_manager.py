class SkillManager:
    def __init__(self, skills, skills_selector, response_selector):
        self.skills = skills
        self.skills_selector = skills_selector
        self.response_selector = response_selector

    def __call__(self, state):
        active_skills = self.skills_selector.select_skills(state)
        skill_responses = self.get_responses(active_skills, state)
        response = self.response_selector.select_response(skill_responses, state)
        return response

    def get_responses(self, active_skills, state):
        """
        Call to skill services, return response.
        Args:
            active_skills:

        Returns:

        """
        ...
