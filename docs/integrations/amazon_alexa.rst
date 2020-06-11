Amazon Alexa integration
========================

DeepPavlov models can be made available for inference via Amazon Alexa. Because of Alexa predominantly
conversational nature (raw text in, raw text out), the best results can be achieved with models with raw text both
in input and output (ODQA, SQuAD, etc.).

Also we **highly** recommend you to study `Alexa skills building basics <https://developer.amazon.com/docs/ask-overviews/build-skills-with-the-alexa-skills-kit.html>`__
and `Alexa Developer console <https://developer.amazon.com/docs/devconsole/about-the-developer-console.html>`__
to make you familiar with main Alexa development concepts and terminology.

Further instructions are given counting on the fact that you are already familiar with them.

The whole integrations process takes two main steps:

1. Skill setup in Amazon Alexa Developer console
2. DeepPavlov skill/model REST service mounting

1. Skill setup
--------------

The main feature of Alexa integration is that Alexa API does not provide  direct ways to pass raw user text to your custom skill.
You will define at least one intent in Developer Console (you will even not be able to compile your skill without one)
and at least one slot (without it you will not be able to pass any user input). Of course, you can not cover infinite
possible user inputs with list of predefined intents and slots. There are to ways to hack it:

**1. AMAZON.SearchQuery slot type**

This hack uses AMAZON.SearchQuery slot type which grabs raw text (speech) user input. Bad news that sample utterance
can not consist only of AMAZON.SearchQuery slot and requires some carrier phrase (one word carrier phrase will work).
So you should define this phrase and restrict your user to use it before or after you query.

Here is JSON config example for Skill Developer console with *"tell"* carrier phrase:

.. code:: json

    {
        "interactionModel": {
            "languageModel": {
                "invocationName": "my beautiful sandbox skill",
                "intents": [
                    {
                        "name": "AMAZON.CancelIntent",
                        "samples": []
                    },
                    {
                        "name": "AMAZON.HelpIntent",
                        "samples": []
                    },
                    {
                        "name": "AMAZON.StopIntent",
                        "samples": []
                    },
                    {
                        "name": "AMAZON.NavigateHomeIntent",
                        "samples": []
                    },
                    {
                        "name": "AskDeepPavlov",
                        "slots": [
                            {
                                "name": "raw_input",
                                "type": "AMAZON.SearchQuery"
                            }
                        ],
                        "samples": [
                            "tell {raw_input}"
                        ]
                    }
                ],
                "types": []
            }
        }
    }

**2. Custom slot type**

This is kind of "black market hack" but it gives the exact result we want. The idea is to use
`custom slot types <https://developer.amazon.com/docs/custom-skills/create-and-edit-custom-slot-types.html>`__.
In our case, we will need only one slot type. We will rely on the fact, that, according the docs values outside the
predefined custom slot values list are still returned if recognized by the spoken language understanding system.
Although input to a custom slot type is weighted towards the values in the list, it is not constrained to just the
items on the list.

The other good news is that custom slot does not require any wrapper words and will grab exact user speech.

So, the recipe is to define only one intent with only one sample utterance which in turn will consist of your only custom slot.
Custom slot values list should consist of several "abracadabra" entries. Here is JSON config example for Skill Developer
console:

.. code:: json

    {
        "interactionModel": {
            "languageModel": {
                "invocationName": "my beautiful sandbox skill",
                "intents": [
                    {
                        "name": "AMAZON.CancelIntent",
                        "samples": []
                    },
                    {
                        "name": "AMAZON.HelpIntent",
                        "samples": []
                    },
                    {
                        "name": "AMAZON.StopIntent",
                        "samples": []
                    },
                    {
                        "name": "AMAZON.NavigateHomeIntent",
                        "samples": []
                    },
                    {
                        "name": "AskDeepPavlov",
                        "slots": [
                            {
                                "name": "raw_input",
                                "type": "GetInput"
                            }
                        ],
                        "samples": [
                            "{raw_input}"
                        ]
                    }
                ],
                "types": [
                    {
                        "name": "GetInput",
                        "values": [
                            {
                                "name": {
                                    "value": "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum"
                                }
                            },
                            {
                                "name": {
                                    "value": "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur"
                                }
                            },
                            {
                                "name": {
                                    "value": "quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat"
                                }
                            },
                            {
                                "name": {
                                    "value": "Ut enim ad minim veniam"
                                }
                            },
                            {
                                "name": {
                                    "value": "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua"
                                }
                            },
                            {
                                "name": {
                                    "value": "Lorem ipsum dolor sit amet, consectetur adipiscing elit"
                                }
                            }
                        ]
                    }
                ]
            }
        }
    }

Please note, that in both cases you should have only one intent with only one slot defined in Alexa Development Console.

2. DeepPavlov skill/model REST service mounting
---------------------------------------------------

Alexa sends request to the https endpoint which was set in the **Endpoint** section of Alexa Development Console.

You should deploy DeepPavlov skill/model REST service on this
endpoint or redirect it to your REST service. Full REST endpoint URL
can be obtained by the swagger ``docs/`` endpoint. We remind you that Alexa requires https endpoint
with valid certificate from CA. `Here is the guide <https://developer.amazon.com/docs/custom-skills/configure-web-service-self-signed-certificate.html>`__
for running custom skill service with self-signed certificates in test mode.

Your intent and slot names defined in Alexa Development Console should be the same with values defined in
DeepPavlov settings file ``deeppavlov/utils/settings/server_config.json``. JSON examples from this guide use default values from
the settings file.

DeepPavlov skill/model can be made available for Amazon Alexa as a REST service by:

.. code:: bash

    python -m deeppavlov alexa <config_path> [--https] [--key <SSL key file path>] \
    [--cert <SSL certificate file path>] [-d] [-p <port_number>]

If you redirect requests to your skills service from some https endpoint, you may want to run it in http mode by
omitting ``--https``, ``--key``, ``--cert`` keys.

Optional ``-d`` key can be provided for dependencies download
before service start.

Optional ``-p`` key can be provided to override the port value from a settings file.
for **each** conversation.

REST service properties (host, port, https options) are provided in ``deeppavlov/utils/settings/server_config.json``. Please note,
that all command line parameters override corresponding config ones.
