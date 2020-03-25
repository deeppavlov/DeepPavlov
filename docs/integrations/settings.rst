DeepPavlov settings
===================

DeepPavlov provides some tools to facilitate its usage (e.g. dialog logging, settings management). This document is aimed to guide you through them.

1. Settings files access and management
---------------------------------------

Most of DeepPavlov settings are located in settings files, which in turn are located in a settings folder. Default settings folder location is ``deeppavlov/utils/settings`` .

You can override a settings directory path by setting the ``DP_SETTINGS_PATH`` environment variable. Missing files will be added automatically when running any deeppavlov script.

You can get current full path to settings directory with ``python -m deeppavlov.settings``.
To reset settings in the current settings directory one can use ``python -m deeppavlov.settings -d``.

2. Dialog logging
-----------------

DeepPavlov supports logging of infered utterances and DeepPavlov model responses. You can manage dialog logging by
editing ``dialog_logger_config.json`` file in a settings directory.

Following dialog logging settings are available:

1. **enabled** (default: ``false``): turns on/off dialog logging for DeepPavlov instance;
2. **log_path** (default: ``~/.deeppavlov/dialog_logs``): sets directory where dialog logs are stored;
3. **logger_name** (default: ``default``): sets subdirectory name for storing dialog logs;
4. **logfile_max_size_kb** (default: ``10240``): sets logfile maximum size in kilobytes. If exceeded, new log file is created;
5. **ensure_ascii** (default: ``false``): If ``true``, converts all non-ASCII symbols in logged content to Unicode code points.

3. Environment variables
------------------------

- **DP_SETTINGS_PATH** — custom path to a directory that contains settings files. It's automatically populated with missing files when running any deeppavlov scripts.
- **DP_SKIP_NLTK_DOWNLOAD** set to ``TRUE`` to prevent automatic downloading of **nltk** packages (``punkt``, ``stopwords``, ``perluniprops``, ``nonbreaking_prefixes``)
