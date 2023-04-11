Installation
============


We support ``Linux`` platform, ``Python 3.6``, ``3.7``, ``3.8``, ``3.9`` and ``3.10``.

.. note::

    * ``Python 3.5`` is not supported!


#. Create a virtual environment:

    .. code:: bash

        python -m venv env

#. Activate the environment:

    * Linux

        .. code:: bash

            source ./env/bin/activate

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
