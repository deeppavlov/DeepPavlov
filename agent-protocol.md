## Формат запроса

```json
{
  "version": 0.9,
  "context": {
    "id": "5c62c80e0110b34e257d0d6f",
    "location": null,
    "date": "2019-02-12 13:20:14.719000",
    "history": {
      "id": "5c62c80e0110b34e257d0d6e",
      "utterances": [
        {
          "id": "5c62c80e0110b34e257d0d68",
          "channel_type": "telegram",
          "text": "Привет!",
          "user_id": "5c62c80e0110b34e257d0d66",
          "annotations": {
            "ner": [
              
            ],
            "coref": [
              
            ],
            "sentiment": [
              
            ]
          }
        },
        {
          "id": "5c62c80e0110b34e257d0d69",
          "active_skill": "chitchat",
          "confidence": 0.85,
          "channel_type": "telegram",
          "text": "Привет, я бот!",
          "user_id": "5c62c80e0110b34e257d0d67"
        },
        {
          "id": "5c62c80e0110b34e257d0d6a",
          "channel_type": "telegram",
          "text": "Как дела?",
          "user_id": "5c62c80e0110b34e257d0d66",
          "annotations": {
            "ner": [
              
            ],
            "coref": [
              
            ],
            "sentiment": [
              
            ]
          }
        },
        {
          "id": "5c62c80e0110b34e257d0d6b",
          "active_skill": "chitchat",
          "confidence": 0.9333,
          "channel_type": "telegram",
          "text": "Хорошо, а у тебя как?",
          "user_id": "5c62c80e0110b34e257d0d67"
        },
        {
          "id": "5c62c80e0110b34e257d0d6c",
          "channel_type": "telegram",
          "text": "И у меня нормально. Когда родился Петр Первый?",
          "user_id": "5c62c80e0110b34e257d0d66",
          "annotations": {
            "ner": [
              
            ],
            "coref": [
              
            ],
            "sentiment": [
              
            ]
          }
        },
        {
          "id": "5c62c80e0110b34e257d0d6d",
          "active_skill": "odqa",
          "confidence": 0.74,
          "channel_type": "telegram",
          "text": "в 1672 году",
          "user_id": "5c62c80e0110b34e257d0d67"
        }
      ]
    },
    "users": [
      {
        "id": "5c62c80e0110b34e257d0d66",
        "user_telegram_id": "47a4aa51-116f-4384-a383-72da90487985",
        "user_type": "human",
        "device_type": null,
        "presonality": null
      },
      {
        "id": "5c62c80e0110b34e257d0d67",
        "user_telegram_id": "2b0fc782-1d3a-4c14-a4c0-83c7ca2702d6",
        "user_type": "bot",
        "device_type": null,
        "presonality": null
      }
    ]
  }
}
```

## Формат ответа

```json
{
  "response": {
    "text": "привет, я бот!",
    "confidence": 0.947,
    "other": {}
  }
}
```
