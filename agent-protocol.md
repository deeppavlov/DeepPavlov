## Формат запроса

```json
{
  "version": 0.9,
  "dialogs": [
    {
      "id": "5c65706b0110b377e17eba41",
      "location": null,
      "utterances": [
        {
          "id": "5c65706b0110b377e17eba39",
          "text": "Привет!",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": [
              
            ],
            "coref": [
              
            ],
            "sentiment": [
              
            ]
          },
          "date_time": "2019-02-14 13:43:07.594000"
        },
        {
          "id": "5c65706b0110b377e17eba3a",
          "active_skill": "chitchat",
          "confidence": 0.85,
          "text": "Привет, я бот!",
          "user_id": "5c65706b0110b377e17eba38",
          "annotations": {
            "ner": [
              
            ],
            "coref": [
              
            ],
            "sentiment": [
              
            ]
          },
          "date_time": "2019-02-14 13:43:07.595000"
        },
        {
          "id": "5c65706b0110b377e17eba3b",
          "text": "Как дела?",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": [
              
            ],
            "coref": [
              
            ],
            "sentiment": [
              
            ]
          },
          "date_time": "2019-02-14 13:43:07.595000"
        },
        {
          "id": "5c65706b0110b377e17eba3c",
          "active_skill": "chitchat",
          "confidence": 0.9333,
          "text": "Хорошо, а у тебя как?",
          "user_id": "5c65706b0110b377e17eba38",
          "annotations": {
            "ner": [
              
            ],
            "coref": [
              
            ],
            "sentiment": [
              
            ]
          },
          "date_time": "2019-02-14 13:43:07.595000"
        },
        {
          "id": "5c65706b0110b377e17eba3d",
          "text": "И у меня нормально. Когда родился Петр Первый?",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": [
              
            ],
            "coref": [
              
            ],
            "sentiment": [
              
            ]
          },
          "date_time": "2019-02-14 13:43:07.595000"
        },
        {
          "id": "5c65706b0110b377e17eba3e",
          "active_skill": "odqa",
          "confidence": 0.74,
          "text": "в 1672 году",
          "user_id": "5c65706b0110b377e17eba38",
          "annotations": {
            "ner": [
              
            ],
            "coref": [
              
            ],
            "sentiment": [
              
            ]
          },
          "date_time": "2019-02-14 13:43:07.595000"
        },
        {
          "id": "5c65706b0110b377e17eba3f",
          "text": "спасибо",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": [
              
            ],
            "coref": [
              
            ],
            "sentiment": [
              
            ]
          },
          "date_time": "2019-02-14 13:43:07.595000"
        }
      ],
      "user": {
        "id": "5c65706b0110b377e17eba37",
        "user_telegram_id": "0801e781-0b76-43fa-9002-fcdc147d35af",
        "user_type": "human",
        "device_type": null,
        "personality": null
      },
      "bot": {
        "id": "5c65706b0110b377e17eba38",
        "user_type": "bot",
        "personality": null
      },
      "channel_type": "telegram"
    },
    {
      "id": "5c65706b0110b377e17eba47",
      "location": null,
      "utterances": [
        {
          "id": "5c65706b0110b377e17eba43",
          "text": "Когда началась Вторая Мировая?",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": [
              
            ],
            "coref": [
              
            ],
            "sentiment": [
              
            ]
          },
          "date_time": "2019-02-14 13:43:07.601000"
        },
        {
          "id": "5c65706b0110b377e17eba44",
          "active_skill": "odqa",
          "confidence": 0.99,
          "text": "1939",
          "user_id": "5c65706b0110b377e17eba38",
          "annotations": {
            "ner": [
              
            ],
            "coref": [
              
            ],
            "sentiment": [
              
            ]
          },
          "date_time": "2019-02-14 13:43:07.601000"
        },
        {
          "id": "5c65706b0110b377e17eba45",
          "text": "Спасибо, бот!",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": [
              
            ],
            "coref": [
              
            ],
            "sentiment": [
              
            ]
          },
          "date_time": "2019-02-14 13:43:07.601000"
        }
      ],
      "user": {
        "id": "5c65706b0110b377e17eba42",
        "user_telegram_id": "a27a94b6-2b9d-4802-8eb6-6581e6f8cd8c",
        "user_type": "human",
        "device_type": null,
        "personality": null
      },
      "bot": {
        "id": "5c65706b0110b377e17eba38",
        "user_type": "bot",
        "personality": null
      },
      "channel_type": "telegram"
    }
  ]
}
```

## Формат ответа

```json
{
  "responses": [
    {
      "text": "привет, я бот!",
      "confidence": 0.947,
      "other": {}
    },
    {
      "text": "как дела?",
      "confidence": 0.3333,
      "other": {}
    }
  ]
}
```
