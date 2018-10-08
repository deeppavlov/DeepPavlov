Yandex Alice integration
========================


Different parts of DeepPavlov library could be launched as skills for Alice by Yandex.

Configure host, port, model endpoint, GET request arguments in ``utils/server_config.json`` or see default values there.

Use your own certificate for HTTPS if you have; otherwise, generate self-signed one like that:

::

    openssl req -new -newkey rsa:4096 -days 365 -nodes -x509 -subj "/CN=MY_DOMAIN_OR_IP" -keyout my.key -out my.crt

Then run

::

    python -m deeppavlov riseapi --api-mode alice --https --key my.key --cert my.crt  <config_path> [-d]


Optional ``-d`` key is for dependencies download before service start.

Now set up and test your dialog (https://dialogs.yandex.ru/developer/). Detailed documentation of the platform could be
found on https://tech.yandex.ru/dialogs/alice/doc/about-docpage/, while other library options described in
:doc:`REST API </devguides/rest_api>` section.