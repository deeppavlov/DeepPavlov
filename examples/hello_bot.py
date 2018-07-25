# This is "Hello world!" example of simple bot implemented in DeepPavlov.
#
# Imports key components to build HelloBot. 
from deeppavlov.core.agent import Agent, HighestConfidenceSelector
from deeppavlov.skills.pattern_matching_skill import PatternMatchingSkill

# Creates skills as pre-defined responses for a user's input containing specific keywords. 
# Every skill returns response and confidence.
hello = PatternMatchingSkill(responses=['Hello world! :)'], patterns=["hi", "hello", "good day"])
bye = PatternMatchingSkill(['Goodbye world! :(', 'See you around.'], ["bye", "chao", "see you"])
fallback = PatternMatchingSkill(["I don't understand, sorry :/", 'I can say "Hello world!" 8)'])

# Agent executes skills and then takes response from the skill with the highest confidence.
HelloBot = Agent([hello, bye, fallback], skills_selector=HighestConfidenceSelector())

# Give the floor to the HelloBot!
print(HelloBot(['Hello!', 'Boo...', 'Bye.']))