# Morphological tagging

## Task description

Morphological tagging consists in assigning labels, describing word morphology, to a pre-tokenized sequence of words.
In the most simple case these labels are just part-of-speech (POS) tags, hence in earlier times of NLP the task was
often referred as POS-tagging. The refined version of the problem which we solve here performs more fine-grained 
classification, also detecting the values of other morphological features, such as case, gender and number for nouns,
mood, tense, etc. for verbs and so on. Morphological tagging is a stage of common NLP pipeline, it generates useful
features for further tasks such as syntactic parsing, named entity recognition or machine translation.