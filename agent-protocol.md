## Формат запроса

```json
{
  "user": {
    "id": 1,
    "name": "",
    "location": "moscow",
    "device_name": "iphone"
  },
  "dialog": {
    "id": 15,
    "context": [
      {
        "utt_id": 6,
        "utt_text": "привет",
        "type": "human",
        "annotations": {
          "ner": [{}],
          "coref": [{}],
          "sentiment": [{}]
        },
        "confidence": 1.0
      },
      {
        "utt_id": 7,
        "utt_text": "привет, я бот!",
        "type": "bot",
        "annotations": {
          "ner": [{}],
          "coref": [{}],
          "sentiment": [{}]
        },
        "confidence": 0.952
      },
      {
        "utt_id": 8,
        "utt_text": "привет, бот",
        "type": "human",
        "annotations": {
          "ner": [{}],
          "coref": [{}],
          "sentiment": [{}]
        },
        "confidence": 1.0
      }
    ],
    "state": {
      "active_skill_history": ["chitchat", "chitchat", "odqa", "odqa", "chitchat"],
      "active_skill": "chitchat",
      "active_skill_duration": 1,
      "dialog_duration": 6
    }
  }
}
```

## Формат ответа

```json
{
  "response": {
    "text": "привет, я бот!",
    "confidence": 0.947
  }
}
```
