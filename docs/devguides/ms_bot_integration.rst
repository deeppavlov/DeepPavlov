Microsoft Bot Framework integration
===================================

Each library component or skill can be made available for
inference via Microsoft Bot Framework.

The whole process takes two main steps:

1. Web App Bot setup in Microsoft Azure
2. DeepPavlov skill/component REST service mounting

1. Web App Bot setup
--------------------

1.  Web App Bot setup guide presumes that you already have
    active Microsoft Azure account and logged in to the main Azure dashboard

2.  **Create Web App Bot**:

    2.1 Go to the *All resources* menu.

    2.2 Click *Add*.

    2.3 Type "bot" in the search pane and select *Web App Bot*.

    .. image:: ../_static/ms_bot_framework/01_web_app_bot.png
       :width: 800

    2.4 Press *"Create"* button on the next screen.

    2.5 Select Web App Bot creation settings.

    2.6 Pay attention to the *Pricing tier*, be sure to select free one:
    *F0 (10K Premium Messages)*.

    2.7 Press *"Create"* button.

    .. image:: ../_static/ms_bot_framework/02_web_app_bot_settings.png
       :width: 800

    2.8 Navigate to your bot control dashboard.

    .. image:: ../_static/ms_bot_framework/03_navigate_to_bot.png
       :width: 1500

3.  **Web App Bot connection configuration**:

    3.1 Navigate to your bot *Settings* menu.

    3.2 Input your DeepPavlov skill/component REST service URL
    to the *Messaging endpoint* pane. Note, that Microsoft Bot
    Framework requires https endpoint with valid certificate from CA.

    3.3 Save somewhere *Microsoft App ID* (*App ID*). To get *App Secret*
    you need to proceed to the *Manage* ling near the *Microsoft App ID* pane.
    You will need both during your DeepPavlov skill/component REST service start.

    .. image:: ../_static/ms_bot_framework/04_bot_settings.png
       :width: 1500

4.  **Web App Bot channels configuration**

    4.1 Microsoft Bot Framework allows your bot to communicate
    to the outer world via different channels. To set up these channels
    navigate to the *Channels* menu, select channel and follow further instructions.

    .. image:: ../_static/ms_bot_framework/05_bot_channels.png
       :width: 1500

2. DeepPavlov skill/component REST service mounting
---------------------------------------------------

MS Bot Framework sends messages from all channels to the https endpoint
which was set in the **Web App Bot connection configuration** section.

You should deploy DeepPavlov skill/component REST service on this
endpoint or terminate it to your REST service. Full REST endpoint URL
can be obtained by the swagger ``apidocs/`` endpoint. We remind you that Microsoft Bot Framework requires https endpoint
with valid certificate from CA.

Each DeepPavlov skill/component can be made available for MS Bot Framework
as a REST service by:

``python -m deeppavlov interactmsbot -i <microsoft_app_id> -s <microsoft_app_secret> <config_path> [-d]``

Use *Microsoft App ID* and *Microsoft App Secret* obtained
in the **Web App Bot connection configuration** section.
Optional ``-d`` key can be provided for dependencies download
before service start.

REST service properties (host, port) are provided in ``utils/server_config.json``.
