MAX_WORKERS = 4

SKILLS = [
    {
        "name": "odqa",
        "url": "http://0.0.0.0:2080/odqa",
        "path": "deeppavlov/configs/dp_assistant/agent_ru_odqa_infer_wiki.json"
    },
    {
        "name": "chitchat",
        "url": "http://0.0.0.0:2081/chitchat",
        "path": "deeppavlov/configs/dp_assistant/agent_ranking_chitchat_2staged_tfidf_smn_v4.json"
    }
]

ANNOTATORS = [
    {
        "name": "ner",
        "url": "http://",
        "path": "deeppavlov/configs/dp_assistant/preproc_ner_rus.json"
    },
    {
        "name": "sentiment",
        "url": "http://",
        "path": ""
    }
]

SKILL_SELECTORS = [
    {
        "name": "chitchat_odqa",
        "utl": "http://0.0.0.0:2082/chitchat_odqa_selector",
        "path": ""
    }
]

RESPONSE_SELECTORS = []

# TODO include Bot?
