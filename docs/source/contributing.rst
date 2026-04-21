
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

- Managing the dependencies:

   Please consider to use ``uv`` to manage the dependencies for development, and
   do not modify the ``pyproject.toml`` file directly.
   If you need to add a new dependency, please do it with ``uv add <package-name>`` and then commit the changes in ``pyproject.toml`` and ``uv.lock``.
   
Since we are using GitHub Actions, only the allowed users can run the test jobs on GitHub Actions.
If you are not an allowed user, please run the test codes locally before making a pull request with pytest.


******************************
How to test your changes locally
******************************

To test your changes locally before making a pull request, please follow these steps:

1. Clone the repository to your local machine if you haven't already:

2. Create a new branch for your changes:

3. Make your changes to the codebase.

4. Install the required dependencies for testing. You can do this by running:

   .. code-block:: bash

      pip install -e .

   Note that we assume you have a virtual environment set up for development.

5. Run the test suite using `pytest`` to ensure that your changes do not break any existing functionality:

   .. code-block:: bash

      pytest -s tests/


   If all tests pass, you can proceed to commit your changes and push your branch to GitHub for a pull request.
   Of course, you can also run specific test files or test cases as needed.

   .. code-block:: bash

      pytest -s tests/test_specific_file.py


   If you are willing to check the coverage of your changes, you can use the following command:

   .. code-block:: bash

      pytest -v --cov=src/nuqulib/ -s --cov-report=html


******************************
LICENSE
******************************

Any contribution from you will be under the MIT License.
