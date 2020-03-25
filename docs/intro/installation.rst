Installation
============


We support ``Linux`` and ``Windows`` platforms, ``Python 3.6`` and ``Python 3.7``.

.. note::

    * ``Python 3.5`` is not supported!

    * installation for ``Windows`` requires ``Git`` for Windows (for example,
      `git <https://git-scm.com/download/win>`_ ), ``Visual Studio 2015/2017``
      with ``C++`` build tools installed!


#. Create a virtual environment:

    .. code:: bash

        python -m venv env

#. Activate the environment:

    * Linux

        .. code:: bash

            source ./env/bin/activate

    * Windows

        .. code:: bash

            .\env\Scripts\activate.bat

#. Install the package inside this virtual environment:

    .. code:: bash

        pip install deeppavlov


Docker Images
-------------

We have built several DeepPavlov based Docker images, which include:

    * DeepPavlov based Jupyter notebook Docker image;
    * Docker images which serve some of our models and allow to access them
      via REST API (``riseapi`` mode).

Here is our `DockerHub repository <https://hub.docker.com/u/deeppavlov/>`_ with
images and deployment instructions.
