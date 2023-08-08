
# Contribution Guide

We are happy that you share your research with us and want to improve our code!

Please follow the steps below to contribute to our project.

If you have any questions or suggestions about the contributing process, please
share them with us on the [forum](https://forum.deeppavlov.ai). Please note that
we do not answer general questions in the github issues interface.

If you are a regular contributor in the DeepPavlov open source project, you can
receive an invitation to one of our events or an opportunity to become a part of
our team.

### How to contribute:

1. Don't start the coding first. You should do a quick search over [existing
issues](https://github.com/deeppavlov/DeepPavlov/issues?q=is%3Aissue) for the
project to see if your suggestion was already discussed or even resolved. If
nothing relevant was found, please create a new one and state what exactly you
would like to implement or fix. You may proceed with coding once someone on our
team accepts your offer.

2. [Fork](https://guides.github.com/activities/forking/) the [DeepPavlov
repository](https://github.com/deeppavlov/DeepPavlov)

3. Checkout the ``dev`` branch from
[the upstream](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/configuring-a-remote-for-a-fork)
as a base for your code:

    ```
    git clone https://github.com/<OWNER>/<REPOSITORY>.git
    cd <REPOSITORY>
    git remote add upstream https://github.com/deeppavlov/DeepPavlov.git
    git fetch upstream
    git checkout -b dev --track upstream/dev
    ```

    afterwards to sync the ``dev`` branch with external updates you can run:
    
    ```
    git checkout dev
    git fetch upstream
    git pull
    ```

4. **Create a new branch and switch** to it. Give it a meaningful name:
    
    ```
    git checkout -b what_my_code_does_branch
    ```

5. **Install DeepPavlov** in editable mode:

    ```
    pip install -e .
    ```

    or

    ```
    pip install -e .[docs,tests]
    ```

    In editable mode changes of the files in the repository directory will
    automatically reflect in your python environment. The last command with
    ``[doc,tests]`` will install additional requirements to build documentation
    and run tests.

6. **Write readable code** and keep it
[PEP8](https://www.python.org/dev/peps/pep-0008/)-ed, **add docstrings** and
keep them consistent with the
[Google Style](http://google.github.io/styleguide/pyguide.html#381-docstrings).
Pay attention that we support typing annotations in every function declaration.

    Accompany your code with **clear comments** to let other people understand
    the flow of your mind.

    If you create new models, refer to the
    [Register your model](https://docs.deeppavlov.ai/en/master/devguides/registry.html)
    section to add it to the DeepPavlov registry of models.

7. We ask you to **add some tests**. This will help us maintain the framework,
and this will help users to understand the feature you introduce. Examples of
implemented tests are available in
[tests/](https://github.com/deeppavlov/DeepPavlov/tree/dev/tests) directory.

8. Please, **update the documentation**, if you committed significant changes to
our code. Make sure that documentation could be built after your changes and
check how it looks using:

    ```
    cd docs
    make html
    ```
    
    The built documentation will be added to ``docs/_build`` directory. Open it
    with your browser.

9. **Commit your changes and push** your feature branch to your GitHub fork:

    ```
    git add my_files
    git commit -m "fix: resolve issue #271"
    git push origin what_my_code_does_branch
    ```
    
    Follow the [semantic commit
    notation](https://seesparkbox.com/foundry/semantic_commit_messages) for the
    name of the commit.

10. Create a new [pull request](https://github.com/deeppavlov/DeepPavlov/pulls)
to get your feature branch merged into dev for others to use. Don't forget to
[reference](https://help.github.com/en/github/writing-on-github/autolinked-references-and-urls)
the GitHub issue associated with your task in the description.

11. **Relax and wait** : )

Some time after that your commit will be assigned to somebody from our team to
check your code.  After a code review and a successful completion of all tests,
your pull request will be approved and pushed into the framework.

If you still have any questions, either on the contribution process or about the
framework itself, please share them with us on our
[forum](https://forum.deeppavlov.ai). Join our official
[Telegram channel](https://t.me/deeppavlov) to get notified about our updates &
news.