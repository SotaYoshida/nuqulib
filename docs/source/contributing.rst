
Contributing to NuQuLib
=======================================

Thank you for considering contributing to this package.

Feedbacks and contributions to NuQuLib are very welcome. These can be: 

- bug reports 

- submitting new functions or a patch to fix bugs

- documentation issues 

- feature requests 

- etc.

For these contributions, it would be nice to let you know a basic policy
(the workflow of code development and LICENSE, shown below) in this project. 

******************************
Workflow of code development
******************************

If you are willing to contribute to this package, please keep in mind the following workflow.

- We use the GitHub to host the package, to track issues/pull requests.

-  Please create a new branch for your work and make a pull request

   -  If you are not familiar with git, please refer to the Git documentation such as
      `Git - Book <https://git-scm.com/book/en/v2>`_.
   -  The ``main`` (default) branch of this repository is protected by
      ‘github branch protection’, so you cannot push directly to the
      main branch.

-  Consistent tests: 

   We use GitHub Actions to run the test codes and to
   build/deploy the document. When some changes are submitted through a
   pull request, the test codes (in ``pytest`` manner) are run to check that
   the changes are not destructive.

   The test jobs are specified in yml files like
   ``.github/workflows/ci.yml`` and one can find the tests code in
   ``tests/`` of the repository.

   If you wish to add a new feature or to fix a bug, please consider to
   add a test code for it. All the classes and functions in the package
   should be accompanied by docstrings. Since we are using sphinx for
   documentation, please follow the docstring format used in the
   existing code or refer to the Sphinx documentation for guidance.

We are using GitHub Actions to manage the tests and documentation build/deploy.
Please note that, for security and to prevent abuse of GitHub Actions,
only the allowed users can run through push or pull request events.
If you are not an allowed user, please run the test codes locally before making a pull request using pytest.


******************************
LICENSE
******************************

Any contribution from you will be under the MIT License.
