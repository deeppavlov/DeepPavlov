
Contribution Guide
=====================

We are happy when you share your research with us and when you improve our
code! There is an easy way to contribute to our project, follow the steps
below. Your commit will be reviewed and added to our dev branch, and will be
added to master branch with the nearest release. Moreover, if you are a
dedicated contributor, you have a chance to get our t-shirt, get invited to
one of our events or even join our team ; )

How to contribute:

#. Don't start the coding first. You should **post an**
   `issue <https://github.com/deepmipt/DeepPavlov/issues>`_ to discuss the
   features you want to add. If our team or other contributors accept your offer
   or give a +1, assign the issue to yourself. Now proceed with coding : )

#. **Write readable code** and keep it 
   `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_-ed, **add docstrings**
   and keep them consistent with the
   `Google Style <http://google.github.io/styleguide/pyguide.html#381-docstrings>`_.
   Pay attention that we support typing annotations in every function
   declaration.

   Accompany code with **clear comments** to let other people understand the
   flow of your mind.

   If you create new models, refer to the :doc:`Register your model
   </devguides/registry>` section to add it to the DeepPavlov registry of
   models.

#. **Clone and/or update** your checked out **copy of DeepPavlov** to ensure
   you have the most recent commits from the master branch:

    .. code:: bash

        git clone git@github.com:deepmipt/DeepPavlov.git
        cd DeepPavlov/
        git fetch origin
        git checkout dev
        git pull

#. **Create a new branch and switch** to it. Give it a meaningful name:

    .. code:: bash

        git checkout -b what_my_code_does_branch

#. We ask you to **add some tests**. This will help us maintain the
   framework, and this will help users to understand the feature you introduce.
   Examples of implemented tests are available in `tests/
   <https://github.com/deepmipt/DeepPavlov/tree/dev/tests>`_
   directory.

#. Please, **update the documentation**, if you committed significant changes
   to our code. 

#. **Commit your changes and push** your feature branch to your GitHub fork.
   Don't forget to reference the GitHub issue associated with your task.
   Squash your commits into a single commit with git's interactive rebase.
   Create a new branch if necessary.

    .. code:: bash

        git add my_files
        git commit -m "fix: resolve issue #271"
        git push origin my_branch

    Follow the `semantic commit notation <https://seesparkbox.com/foundry/semantic_commit_messages>`_
    for the name of the commit.

#. **Create a new pull request** to get your feature branch merged into dev
   for others to use. You’ll first need to ensure your feature branch contains
   the latest changes from dev. 

    .. code:: bash

        # (external contribs): make a new pull request:

        # merge latest dev changes into your feature branch
        git fetch origin
        git checkout dev
        git pull origin dev 
        git checkout my_branch
        git merge dev  # you may need to manually resolve merge conflicts

#. Once your change has been successfully merged, you can **remove the source
   branch** and ensure your local copy is up to date:

    .. code:: bash

        git fetch origin
        git checkout dev
        git pull
        git branch -d my_branch
        git branch -d -r origin/my_branch

#. **Relax and wait** : )

Some time after that your commit will be reassigned to somebody from our team
to check your code. 
If the code is okay and all tests work fine, your commit will be approved and
added to the framework. Your research will become a part of a common big work
and other people will happily use it and thank you :D 

If you still have any questions, either on the contribution process or about
the framework itself, please ask us at our forum `<https://forum.deeppavlov.ai/>`_.
Follow us on Facebook to get news on releases, new features, approved
contributions and resolved issues `<https://www.facebook.com/deepmipt/>`_

