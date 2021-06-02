from typing import List

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