
Contribution Guide
=====================

We are happy when you share your research with us and when you improve our
code! There is an easy way to contribute to our project, follow the steps
below. Your commit will be reviewed and added to our master branch. Moreover,
if you are a dedicated contributor, you have a chance to get our t-shirt,
get invited to one of our events or even join our team ; )

How to contribute:

#. Don't start the coding first. You should **post an issue** to discuss the features
   you want to add. If our team or other contributors accept your offer
   or give a +1, assign the issue to yourself. Now proceed with coding : )

#. **Add clear comments** to each line of your code to let other people understand
   the flow of your mind.

#. **Clone and/or update** your checked out **copy of DeepPavlov** to ensure
   you have the most recent commits from the master branch:

    ::

        git clone 
        cd 
        git fetch origin
        git checkout dev
        git pull

#. **Create a new branch and switch** to it. Give it a meaningful name:

    ::

        git checkout -b what_my_code_does_branch

#. We demand you to **add some unit tests**. This will help us maintain the
   framework, and this will help users to understand the feature you introduce.

#. Please, **update the documentation**, if you committed significant changes
   to our code. 

#. **Commit your changes and push** your feature branch to your GitHub fork.
   Don't forget to reference the GitHub issue associated with your task.
   Squash your commits into a single commit with git's interactive rebase.
   Create a new branch if necessary.

    ::

        git add my_files
        git commit -m "Issue 271"
        git push origin my_branch

#. **Create a new pull request** to get your feature branch merged into master
   for others to use. You’ll first need to ensure your feature branch contains
   the latest changes from master. 

    ::

        # (external contribs): make a new pull request:

        # merge latest master changes into your feature branch
        git fetch origin
        git checkout master
        git pull origin master
        git checkout my_branch
        git merge master  # you may need to manually resolve merge conflicts

#. Once your change has been successfully merged, you can **remove the source
   branch** and ensure your local copy is up to date:

    ::

        git fetch origin
        git checkout master
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
the framework itself, please ask us at our forum `<https://forum.ipavlov.ai/>`_.
Follow us on Facebook to get news on releases, new features, approved
contributions and resolved issues `<https://www.facebook.com/deepmipt/>`_

