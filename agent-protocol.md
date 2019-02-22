## Формат запроса

```json
{
  "version": "0.9.1",
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
            "ner": {
              "tokens": [
                "Привет",
                "!"
              ],
              "tags": [
                "O",
                "O"
              ]
            },
            "coref": [
              
            ],
            "sentiment": "speech"
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
            "ner": {
              "tokens": [
                "Привет",
                ",",
                "я",
                "бот",
                "!"
              ],
              "tags": [
                "O",
                "O",
                "O",
                "O",
                "O"
              ]
            },
            "coref": [
              
            ],
            "sentiment": "speech"
          },
          "date_time": "2019-02-14 13:43:07.595000"
        },
        {
          "id": "5c65706b0110b377e17eba3b",
          "text": "Как дела?",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": {
              "tokens": [
                "Как",
                "дела",
                "?"
              ],
              "tags": [
                "O",
                "O",
                "O"
              ]
            },
            "coref": [
              
            ],
            "sentiment": "speech"
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
            "ner": {
              "tokens": [
                "Хорошо",
                ",",
                "а",
                "у",
                "тебя",
                "как",
                "?"
              ],
              "tags": [
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O"
              ]
            },
            "coref": [
              
            ],
            "sentiment": "speech"
          },
          "date_time": "2019-02-14 13:43:07.595000"
        },
        {
          "id": "5c65706b0110b377e17eba3d",
          "text": "И у меня нормально. Когда родился Петр Первый?",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": {
              "tokens": [
                "И",
                "у",
                "меня",
                "нормально",
                ".",
                "Когда",
                "родился",
                "Петр",
                "Первый",
                "?"
              ],
              "tags": [
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "O",
                "B-PER",
                "I-PER",
                "O"
              ]
            },
            "coref": [
              
            ],
            "sentiment": "neutral"
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
            "ner": {
              "tokens": [
                "в",
                "1672",
                "году"
              ],
              "tags": [
                "O",
                "O",
                "O"
              ]
            },
            "coref": [
              
            ],
            "sentiment": "neutral"
          },
          "date_time": "2019-02-14 13:43:07.595000"
        },
        {
          "id": "5c65706b0110b377e17eba3f",
          "text": "спасибо",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": {
              "tokens": [
                "спасибо"
              ],
              "tags": [
                "O"
              ]
            },
            "coref": [
              
            ],
            "sentiment": "speech"
          },
          "date_time": "2019-02-14 13:43:07.595000"
        }
      ],
      "user": {
        "id": "5c65706b0110b377e17eba37",
        "user_telegram_id": "0801e781-0b76-43fa-9002-fcdc147d35af",
        "user_type": "human",
        "device_type": null,
        "personality": null,
        "profile": {
          "name": "Джо Неуловимый",
          "gender": "male",
          "birthdate": "2000-02-15",
          "location": null,
          "home_coordinates": null,
          "work_coordinates": null,
          "occupation": "data scientist",
          "inc ome_per_year": 1000000000.0
        }
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
            "ner": {
              "tokens": [
                "Когда",
                "началась",
                "Вторая",
                "Мировая",
                "?"
              ],
              "tags": [
                "O",
                "O",
                "O",
                "O",
                "O"
              ]
            },
            "coref": [
              
            ],
            "sentiment": "neutral"
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
            "ner": {
              "tokens": [
                "1939"
              ],
              "tags": [
                "O"
              ]
            },
            "coref": [
              
            ],
            "sentiment": "neutral"
          },
          "date_time": "2019-02-14 13:43:07.601000"
        },
        {
          "id": "5c65706b0110b377e17eba45",
          "text": "Спасибо, бот!",
          "user_id": "5c65706b0110b377e17eba37",
          "annotations": {
            "ner": {
              "tokens": [
                "Спасибо",
                ",",
                "бот",
                "!"
              ],
              "tags": [
                "O",
                "O",
                "O",
                "O"
              ]
            },
            "coref": [
              
            ],
            "sentiment": "speech"
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

## Базовый формат ответа скила

```json
{
  "responses": [
    {
      "text": "привет, я бот!",
      "confidence": 0.947,
      "skill_name": "chitchat"
    },
    {
      "text": "как дела?",
      "confidence": 0.3333,
      "skill_name": "chitchat"
    }
  ]
}
```
