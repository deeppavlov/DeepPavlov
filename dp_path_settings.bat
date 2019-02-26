SET PATH_TO_VENV=C:\Users\alpha
SET BASE_PATH=E:\nvb\DeepPavlov
REM SET BASE_PATH=E:\alpha\empBot

set PATH=%PATH_TO_VENV%\AppData\Local\conda\conda\envs\dp_venv\Scripts;%PATH_TO_VENV%\AppData\Local\conda\conda\envs\dp_venv;%PATH%  
SET PYTHONPATH=%BASE_PATH%\FictionEmpatBot;%BASE_PATH%\FictionEmpatBot\queries;%BASE_PATH%\FictionEmpatBot\intention_classifier;%BASE_PATH%\FictionEmpatBot\answers_generator;%PYTHONPATH%

