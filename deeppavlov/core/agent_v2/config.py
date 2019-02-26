
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
        "path": "deeppavlov/configs/dp_assistant/ranking_chitchat_2staged_tfidf_smn_v4.json"
    }
]
