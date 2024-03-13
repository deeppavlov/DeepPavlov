Installation
============

DeepPavlov supports **Linux**, **Windows 10+** (through WSL/WSL2), **MacOS** (Big Sur+) platforms, **Python 3.6-3.11**.
Depending on the model used, you may need from 4 to 16 GB RAM.

Install with pip
~~~~~~~~~~~~~~~~

You should install DeepPavlov in a `virtual environment <https://docs.python.org/3/library/venv.html>`_. If you’re
unfamiliar with Python virtual environments, take a look at this
`guide <https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/>`_. A virtual
environment makes it easier to manage different projects, and avoid compatibility issues between dependencies.

#. Create a virtual environment:

    .. code:: bash

        python -m venv env

#. Activate the virtual environment on Linux (`source` could be replaced with `.`):

    .. code:: bash

        source env/bin/activate

#. Install DeepPavlov inside this virtual environment:

    .. code:: bash

        pip install deeppavlov

Install from source
~~~~~~~~~~~~~~~~~~~

Install DeepPavlov **dev** branch from source with the following command:

    .. code:: bash

        pip install git+http://github.com/deeppavlov/DeepPavlov@dev

This command installs the bleeding edge dev version rather than the latest release version. The dev version is useful
for staying up-to-date with the latest developments. For instance, if a bug has been fixed since the last release but
a new release hasn’t been rolled out yet. However, this means the dev version may not always be stable.

Editable install
~~~~~~~~~~~~~~~~

You will need an editable install if you want to make changes in the DeepPavlov source code that immediately take place
without requiring a new installation.

Clone the repository and install DeepPavlov with the following commands:

    .. code:: bash

        git clone http://github.com/deeppavlov/DeepPavlov.git
        pip install -e DeepPavlov

Docker Images
~~~~~~~~~~~~~

We have built several DeepPavlov based Docker images, which include:

    * DeepPavlov based Jupyter notebook Docker image;
    * Docker images which serve some of our models and allow to access them
      via REST API (:doc:`riseapi </integrations/rest_api>` mode).

Here is our `DockerHub repository <https://hub.docker.com/u/deeppavlov/>`_ with
images and deployment instructions.
