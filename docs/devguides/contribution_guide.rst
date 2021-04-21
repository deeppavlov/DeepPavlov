
Contribution Guide
=====================

We are happy that you share your research with us and want to improve our code!

Please follow the steps below to contribute to our project.

If you have any questions or suggestions about the contributing process,
please share them with us on the `forum <https://forum.deeppavlov.ai>`_.
Please note that we do not answer general questions in the github issues interface.

If you are a regular contributor in the DeepPavlov open source project,
you can receive an invitation to one of our events or an opportunity to become a part of our team.

How to contribute:

#. Don't start the coding first.
   You should do a quick search over `existing issues <https://github.com/deepmipt/DeepPavlov/issues?q=is%3Aissue>`_
   for the project to see if your suggestion was already discussed or even resolved.
   If nothing relevant was found, please create a new one and state what exactly you would like
   to implement or fix.
   You may proceed with coding once someone on our team accepts your offer.

#. `Fork <https://guides.github.com/activities/forking/>`_ the
   `DeepPavlov repository <https://github.com/deepmipt/DeepPavlov>`_

#. Checkout the ``dev`` branch from
   `the upstream <https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/configuring-a-remote-for-a-fork>`_
   as a base for your code:

    .. code:: bash

        git clone https://github.com/<OWNER>/<REPOSITORY>.git
        cd <REPOSITORY>
        git remote add upstream https://github.com/deepmipt/DeepPavlov.git
        git fetch upstream
        git checkout -b dev --track upstream/dev

   afterwards to sync the ``dev`` branch with external updates you can run:

    .. code:: bash

        git checkout dev
        git fetch upstream
        git pull

#. **Create a new branch and switch** to it. Give it a meaningful name:

    .. code:: bash

        git checkout -b what_my_code_does_branch

#. **Write readable code** and keep it
   `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_-ed, **add docstrings**
   and keep them consistent with the
   `Google Style <http://google.github.io/styleguide/pyguide.html#381-docstrings>`_.
   Pay attention that we support typing annotations in every function
   declaration.

   Accompany your code with **clear comments** to let other people understand the
   flow of your mind.

   If you create new models, refer to the :doc:`Register your model
   </devguides/registry>` section to add it to the DeepPavlov registry of
   models.

#. We ask you to **add some tests**. This will help us maintain the
   framework, and this will help users to understand the feature you introduce.
   Examples of implemented tests are available in `tests/
   <https://github.com/deepmipt/DeepPavlov/tree/dev/tests>`_
   directory.

#. Please, **update the documentation**, if you committed significant changes
   to our code. 

#. **Commit your changes and push** your feature branch to your GitHub fork:

    .. code:: bash

        git add my_files
        git commit -m "fix: resolve issue #271"
        git push origin what_my_code_does_branch

    Follow the `semantic commit notation <https://seesparkbox.com/foundry/semantic_commit_messages>`_
    for the name of the commit.

#. Create a new `pull request <https://github.com/deepmipt/DeepPavlov/pulls>`_
   to get your feature branch merged into dev for others to use.
   Don't forget to `reference <https://help.github.com/en/github/writing-on-github/autolinked-references-and-urls>`_
   the GitHub issue associated with your task in the description.

#. **Relax and wait** : )

Some time after that your commit will be assigned to somebody from our team
to check your code. 
After a code review and a successful completion of all tests, your pull request will be approved and
pushed into the framework.

If you still have any questions, either on the contribution process or about
the framework itself, please share them with us on our `forum <https://forum.deeppavlov.ai>`_.
Join our official `Telegram channel <https://t.me/deeppavlov>`_ to get notified about our updates & news.
