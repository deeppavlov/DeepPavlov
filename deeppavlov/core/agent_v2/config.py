from deeppavlov import configs


MAX_WORKERS = 4

SKILLS = [
    {
        "name": "odqa",
        "url": "http://0.0.0.0:2080/odqa",
        "path": configs.dp_assistant.agent_ru_odqa_infer_wiki
    },
    {
        "name": "chitchat",
        "url": "http://0.0.0.0:2081/chitchat",
        "path": configs.dp_assistant.agent_ranking_chitchat_2staged_tfidf_smn_v4
    }
]

ANNOTATORS = [
    {
        "name": "ner",
        "url": "http://0.0.0.0:2083/ner_rus",
        "path": configs.dp_assistant.preproc_ner_rus
    },
    {
        "name": "sentiment",
        "url": "http://0.0.0.0:2084/rusentiment",
        "path": configs.dp_assistant.preproc_rusentiment
    }
]

SKILL_SELECTORS = [
    {
        "name": "chitchat_odqa",
        "url": "http://0.0.0.0:2082/chitchat_odqa_selector",
        "path": configs.dp_assistant.sselector_chitchat_odqa
    }
]

RESPONSE_SELECTORS = []

# TODO include Bot?
