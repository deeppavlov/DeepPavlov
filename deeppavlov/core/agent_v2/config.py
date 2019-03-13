from deeppavlov import configs


MAX_WORKERS = 4

SKILLS = [
    {
        "name": "odqa",
        "url": "http://0.0.0.0:2080/odqa",
        "path": configs.dp_assistant.agent_ru_odqa_retr_noans_rubert_infer
    },
    {
        "name": "chitchat",
        "url": "http://0.0.0.0:2081/chitchat",
        "path": configs.dp_assistant.agent_ranking_chitchat_2staged_tfidf_smn_v4,
        "env": {
            "CUDA_VISIBLE_DEVICES": ""
        }
    },
    {
        "name": "hellobot",
        "url": "http://127.0.0.1:2085/ruler_call/",
        "path": None
    },
    {
        "name": "sberchat",
        "url": "http://23.102.48.212:8443/api/",
        "path": None
    },
    {
        "name": "gen_chitchat",
        "url": "http://0.0.0.0:2086/gen_chitchat",
        "path": configs.dp_assistant.agent_transformer_chit_chat_40k_v01_1_20
    },
    {
        "name": "kbqa",
        "url": "http://0.0.0.0:2087/kbqa",
        "path": configs.dp_assistant.agent_kbqa_rus
    }
]

ANNOTATORS = [
    {
        "name": "ner",
        "url": "http://0.0.0.0:2083/ner_rus",
        "path": configs.dp_assistant.preproc_ner_rus,
        "env": {
            "CUDA_VISIBLE_DEVICES": ""
        }
    },
    {
        "name": "sentiment",
        "url": "http://0.0.0.0:2084/rusentiment",
        "path": configs.dp_assistant.preproc_rusentiment,
        "env": {
            "CUDA_VISIBLE_DEVICES": ""
        }
    }
]

SKILL_SELECTORS = [
    {
        "name": "chitchat_odqa",
        "url": "http://0.0.0.0:2082/chitchat_odqa_selector",
        "path": configs.dp_assistant.sselector_chitchat_odqa,
        "env": {
            "CUDA_VISIBLE_DEVICES": ""
        }
    }
]

RESPONSE_SELECTORS = []

# TODO include Bot?
