DeepPavlov settings
===================

DeepPavlov provides some tools to facilitate its usage (e.g. dialog logging, settings management). This document is aimed to guide you through them.

1. Settings files access and management
---------------------------------------

Most of DeepPavlov settings are located in settings files, which in turn are located in settings folder. Default settings folder location is ``utils/settings`` .

You can get full path to settings folder with ``python -m deeppavlov.settings settings``. Also you can move it with with ``python -m deeppavlov.settings settings -p <new/configs/dir/path>`` (all your configuration settings will be preserved) or move it to default location (``utils/settings``) with ``python -m deeppavlov.settings settings -d`` (all your configuration settings will be RESET to default ones).

2. Dialog logging
-----------------

DeepPavlov supports logging of dialogs carried by Agent or ``riseapi`` instances. You can manage dialog logging by editing ``dialog_logger_config.json`` settings file in settings directory

Following dialog logging settings are available:

1. **enabled** (default: ``false``): turns on/off dialog logging for DeepPavlov instance;
2. **log_path** (default: ``~/.deeppavlov/dialog_logs``): sets directory where dialog logs are stored;
3. **agent_name** (default: ``dp_agent``): sets subdirectory name for storing dialog logs;
4. **logfile_max_size_kb** (default: ``10240``): sets logfile maximum size in kilobytes. If exceeded, new log file is created;
5. **ensure_ascii** (default: ``false``): If ``true``, converts all non-ASCII symbols in logged content to Unicode code points.

3. Environment variables
------------------------

- **DP_SKIP_NLTK_DOWNLOAD** set to ``TRUE`` to prevent automatic downloading of **nltk** packages (``punkt``, ``stopwords``, ``perluniprops``, ``nonbreaking_prefixes``)
