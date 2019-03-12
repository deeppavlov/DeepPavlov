Deployment
==========
1. Install project requirements.
2. Configure and run `Mongo DB` in [connection.py](connection.py)
3. Configure and run skills, skill selectors, response selectors and annotators servers in [config.py](config.py).

    * don't change ``name`` fields
    * all configs where ``path`` is not ``None`` are DeepPavlov's configs and they can run in `skill` riseapi mode: 
    ```bash
    python -m deeppavlov riseapi --api-mode skill <CONFIG_REL_PATH> --port <PORT> --endpoint <ENDPOINT>
    ```
    * ``hellobot`` skill is not DeepPavlov's, it has a separate [instruction]( https://github.com/acriptis/dj_bot/blob/master/hello_bot/README.md#deployment) how to deploy it.
    * ``odqa`` skill should run on GPU. For other DeepPavlov skills GPU is not critical.
4. Configure `TELEGRAM_TOKEN` and `TELEGRAM_PROXY` environment variables.
5. Run [run.py](run.py). Conversation with the Agent should become available via Telegram.