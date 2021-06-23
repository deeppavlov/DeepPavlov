from typing import List
from deeppavlov.core.common.file import read_yaml

USER = "usr"
SYSTEM = "sys"

class Turn:
    def __init__(self, turn_description: str, whose_turn: str):
        self.turn_description = turn_description
        self.whose_turn = whose_turn

    def is_user_turn(self):
        return self.whose_turn == USER

    def is_system_turn(self):
        return self.whose_turn == SYSTEM

class Story:
    def __init__(self, title, turns: List[Turn] = None):

        self.title = title
        if turns is None:
            turns = list()
        self.turns = turns.copy()


class Stories:
    def __init__(self):
        self.stories: List[Story] = list()
        self.lines = None

    @classmethod
    def from_stories_lines_md(cls, lines: List[str], fmt="md"):
        if fmt != "md":
            raise Exception(f"Support of fmt {fmt} is not implemented")

        stories = cls()
        lines = [line.strip() for line in lines if line.strip()]
        stories.lines = lines.copy()
        for line in lines:
            if line.startswith('#'):
                # #... marks the beginning of new story
                curr_story_title = line.strip('#')
                curr_story = Story(curr_story_title)
                stories.stories.append(curr_story)
            if line.startswith('*'):
                line_content = line.lstrip('*').strip()
                # noinspection PyUnboundLocalVariable
                curr_story.turns.append(Turn(line_content, USER))
            elif line.startswith('-'):
                line_content = line.strip('-').strip()
                # noinspection PyUnboundLocalVariable
                curr_story.turns.append(Turn(line_content, SYSTEM))
        return stories

    @classmethod
    def from_stories_lines_yml(cls, lines: List[str], fmt="yml"):
        lines_text = '\n'.join(lines)
        stories_yml = read_yaml(lines_text)
        stories_lines = []
        for story in stories_yml.get("stories", []):
            story_title = story.get("story", 'todo')
            stories_lines.append(f"# {story_title}")
            for step in story.get("steps", []):
                is_usr_step = "intent" in step.keys()
                is_sys_step = "action" in step.keys()
                if is_usr_step:
                    curr_story_line = step["intent"]
                    stories_lines.append(f"* {curr_story_line}")
                if is_sys_step:
                    curr_story_line = step["action"]
                    stories_lines.append(f"- {curr_story_line}")

        return cls.from_stories_lines_md(stories_lines)

    @classmethod
    def from_stories_lines(cls, lines: List[str]):
        try:
            lines_text = '\n'.join(lines)
            read_yaml(lines_text)
            is_yaml = True
            is_md = False
        except:
            is_yaml = False
            is_md = True

        if is_yaml:
            return cls.from_stories_lines_yml(lines)
        if is_md:
            return cls.from_stories_lines_md(lines)