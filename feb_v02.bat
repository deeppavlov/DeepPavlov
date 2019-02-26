REM set BASE_PATH = E:\nvb
REM set BASE_PATH =E:\alpha\empBot
set PYTHONPATH=E:\alpha\empBot\DeepPavlov\FictionEmpatBot;E:\alpha\empBot\DeepPavlov\FictionEmpatBot\queries;E:\alpha\empBot\DeepPavlov\FictionEmpatBot\intention_classifier;E:\alpha\empBot\FictionEmpatBot\answers_generator;%PYTHONPATH%

python -m deeppavlov interactbot deeppavlov/configs/feb/feb_v01.json -d
