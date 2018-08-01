This is "Hello, World!" example of simple bot implemented in DeepPavlov
=======================================================================

Import key components to build HelloBot.

.. code:: python

    from deeppavlov.core.agent import Agent, HighestConfidenceSelector
    from deeppavlov.skills.pattern_matching_skill import PatternMatchingSkill

Create skills as pre-defined responses for a user's input containing
specific keywords. Every skill returns response and confidence.

.. code:: python

    hello = PatternMatchingSkill(responses=['Hello world! :)'], patterns=["hi", "hello", "good day"])
    bye = PatternMatchingSkill(['Goodbye world! :(', 'See you around.'], ["bye", "chao", "see you"])
    fallback = PatternMatchingSkill(["I don't understand, sorry :/", 'I can say "Hello world!" 8)'])

Agent executes skills and then takes response from the skill with the
highest confidence.

.. code:: python

    HelloBot = Agent([hello, bye, fallback], skills_selector=HighestConfidenceSelector())

Give the floor to the HelloBot!

.. code:: python

    print(HelloBot(['Hello!', 'Boo...', 'Bye.']))

`Jupyter notebook with HelloBot example. <https://github.com/deepmipt/DeepPavlov/blob/master/examples/hello_bot.ipynb>`__
