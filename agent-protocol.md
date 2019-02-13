## Формат запроса

```json
{
  "version": 0.9,
  "batch": [
    {
      "id": "5c62f7330110b36bdd1dc5df",
      "location": null,
      "history": {
        "utterances": [
          {
            "id": "5c62f7330110b36bdd1dc5d7",
            "text": "Привет!",
            "user_id": "5c62f7330110b36bdd1dc5d5",
            "annotations": {
              "ner": [
                
              ],
              "coref": [
                
              ],
              "sentiment": [
                
              ]
            },
            "date": "2019-02-12 16:41:23.142000"
          },
          {
            "id": "5c62f7330110b36bdd1dc5d8",
            "active_skill": "chitchat",
            "confidence": 0.85,
            "text": "Привет, я бот!",
            "user_id": "5c62f7330110b36bdd1dc5d6",
            "annotations": {
              "ner": [
                
              ],
              "coref": [
                
              ],
              "sentiment": [
                
              ]
            },
            "date": "2019-02-12 16:41:23.142000"
          },
          {
            "id": "5c62f7330110b36bdd1dc5d9",
            "text": "Как дела?",
            "user_id": "5c62f7330110b36bdd1dc5d5",
            "annotations": {
              "ner": [
                
              ],
              "coref": [
                
              ],
              "sentiment": [
                
              ]
            },
            "date": "2019-02-12 16:41:23.142000"
          },
          {
            "id": "5c62f7330110b36bdd1dc5da",
            "active_skill": "chitchat",
            "confidence": 0.9333,
            "text": "Хорошо, а у тебя как?",
            "user_id": "5c62f7330110b36bdd1dc5d6",
            "annotations": {
              "ner": [
                
              ],
              "coref": [
                
              ],
              "sentiment": [
                
              ]
            },
            "date": "2019-02-12 16:41:23.142000"
          },
          {
            "id": "5c62f7330110b36bdd1dc5db",
            "text": "И у меня нормально. Когда родился Петр Первый?",
            "user_id": "5c62f7330110b36bdd1dc5d5",
            "annotations": {
              "ner": [
                
              ],
              "coref": [
                
              ],
              "sentiment": [
                
              ]
            },
            "date": "2019-02-12 16:41:23.143000"
          },
          {
            "id": "5c62f7330110b36bdd1dc5dc",
            "active_skill": "odqa",
            "confidence": 0.74,
            "text": "в 1672 году",
            "user_id": "5c62f7330110b36bdd1dc5d6",
            "annotations": {
              "ner": [
                
              ],
              "coref": [
                
              ],
              "sentiment": [
                
              ]
            },
            "date": "2019-02-12 16:41:23.143000"
          },
          {
            "id": "5c62f7330110b36bdd1dc5dd",
            "text": "спасибо",
            "user_id": "5c62f7330110b36bdd1dc5d5",
            "annotations": {
              "ner": [
                
              ],
              "coref": [
                
              ],
              "sentiment": [
                
              ]
            },
            "date": "2019-02-12 16:41:23.143000"
          }
        ]
      },
      "user": {
        "id": "5c62f7330110b36bdd1dc5d5",
        "user_telegram_id": "44d279ea-62ab-4c71-9adb-ed69143c12eb",
        "user_type": "human",
        "device_type": null,
        "personality": null
      },
      "bot": {
        "id": "5c62f7330110b36bdd1dc5d6",
        "user_telegram_id": "56f1d5b2-db1a-4128-993d-6cd1bc1b938f",
        "user_type": "bot",
        "device_type": null,
        "personality": null
      },
      "channel_type": "telegram"
    },
    {
      "id": "5c62f7330110b36bdd1dc5e4",
      "location": null,
      "history": {
        "utterances": [
          {
            "id": "5c62f7330110b36bdd1dc5e0",
            "text": "Когда началась Вторая Мировая?",
            "user_id": "5c62f7330110b36bdd1dc5d5",
            "annotations": {
              "ner": [
                
              ],
              "coref": [
                
              ],
              "sentiment": [
                
              ]
            },
            "date": "2019-02-12 16:41:23.158000"
          },
          {
            "id": "5c62f7330110b36bdd1dc5e1",
            "active_skill": "odqa",
            "confidence": 0.99,
            "text": "1939",
            "user_id": "5c62f7330110b36bdd1dc5d6",
            "annotations": {
              "ner": [
                
              ],
              "coref": [
                
              ],
              "sentiment": [
                
              ]
            },
            "date": "2019-02-12 16:41:23.158000"
          },
          {
            "id": "5c62f7330110b36bdd1dc5e2",
            "text": "Спасибо, бот!",
            "user_id": "5c62f7330110b36bdd1dc5d5",
            "annotations": {
              "ner": [
                
              ],
              "coref": [
                
              ],
              "sentiment": [
                
              ]
            },
            "date": "2019-02-12 16:41:23.158000"
          }
        ]
      },
      "user": {
        "id": "5c62f7330110b36bdd1dc5d5",
        "user_telegram_id": "44d279ea-62ab-4c71-9adb-ed69143c12eb",
        "user_type": "human",
        "device_type": null,
        "personality": null
      },
      "bot": {
        "id": "5c62f7330110b36bdd1dc5d6",
        "user_telegram_id": "56f1d5b2-db1a-4128-993d-6cd1bc1b938f",
        "user_type": "bot",
        "device_type": null,
        "personality": null
      },
      "channel_type": "facebook"
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
