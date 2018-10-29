DeepPavlov settings
===================

DeepPavlov provides some tools to facilitate its usage (e.g. dialog logging, settings management). This document is aimed guide you through them.

1. Settings files access and management
---------------------------------------

Most of DeepPavlov settings are located in settings files, which in turn are located in settings folder.

You can get full path to settings folder with `python -m deeppavlov.settings settings`. Also you can move it with with `python -m deeppavlov.settings settings -p <new/configs/dir/path>` (all your configuration settings will be preserved) or move it to default location (`utils/settings`) with `python -m deeppavlov.settings settings -d` (all your configuration settings will be RESET to default ones).

2. Dialog logging
-----------------

DeepPavlov supports logging of dialogs carried by Agent or `riseapi` instances. You can manage dialog logging by editing `dialog_logger_config.json` settings file in settings directory

Following dialog logging settings are available:

1. **enabled**: turns on/off dialog logging for DeepPavlov instance;
2. **log_path**: sets directory where dialog logs are stored;
3. **agent_name**: sets subdirectory name for storing dialog logs;
4. **logfile_max_size_kb**: sets logfile maximum size. If exceeded, new log file is created.