Installation
============

0. Currently we support ``Linux`` and ``Windows`` platforms and ``Python 3.6``
    * ``Python 3.5`` is not supported!
    * ``Windows`` platform requires ``Git`` for Windows (for example, `git <https://git-scm.com/download/win>`__ ), ``Visual Studio 2015/2017`` with ``C++`` build tools installed!

1. Create a virtual environment with ``Python 3.6``:

    .. code:: bash

        virtualenv env

2. Activate the environment:

    * Linux

        .. code:: bash

            source ./env/bin/activate

    * Windows

        .. code:: bash

            .\env\Scripts\activate.bat

3. Install the package inside this virtual environment:

    .. code:: bash

        pip install deeppavlov

