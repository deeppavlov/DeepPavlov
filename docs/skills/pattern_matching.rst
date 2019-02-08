Pattern Matching Skill
======================

A :doc:`basic skill implementation</apiref/skills/pattern_matching_skill>` that will always respond with
one of predefined responses chosen at random. Skill's confidence equals ``1`` for incoming utterances that match any
of predefined patterns or ``0`` for utterances that do not. If no patterns were defined for a skill, its confidence will
always be equal to ``0.5``.

Its usage example can be found in the :doc:`Hello bot! </intro/hello_bot>` tutorial.
