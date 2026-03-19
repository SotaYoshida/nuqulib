Nuclear Quantum computing Library
=======================================


.. toctree::
   :hidden:

   contributing
   nuqulib

**********
Features
**********

Nuclear Quantum computing Library (NuQlib) provides a set of tools for quantum simulations of nuclear systems.
One of the main motivations to develop NuQuLib was to provide a library for quantum computing
that can be used in conjunction with existing nuclear physics codes.

One can use some interaction formats adopted in e.g. `NuHamil <https://github.com/Takayuki-Miyagi/NuHamil-public>`_ and
`KSHELL <https://sites.google.com/alumni.tsukuba.ac.jp/kshell-nuclear/>`_,
and try a subspace method called QSCI (a.k.a. SQD) through `NuclearToolkit.jl <https://github.com/SotaYoshida/NuclearToolkit.jl>`_.


*************************
Installation
*************************

A recommended environment to set up the dependencies and run NuQuLib is one like `uv`, which is a tool for managing Python virtual environments and dependencies.
You can create a new environment and install the dependencies using the following commands:

.. code-block:: bash
   :caption: uv

   uv init --package your_project

Then you will get directory structure like this:

.. code-block:: bash

   your_project/
   ├── .git
   ├── .gitignore
   ├── .python-version
   ├── pyproject.toml   
   └── README.md
   ├── src
   
In the root directory of NuQuLib (or your project), you can install the dependencies using the following command:

.. code-block:: bash
   :caption: uv

   uv sync

If you are using `pip`, you may install the dependencies using the following command:

.. code-block:: bash
   :caption: pip

   pip install -e /path/to/nuqulib

However, compatibility with `pip` is not guaranteed, and we recommend using `uv` instead.

When a GPU is available, you need to remove `qiskit-aer` installed by default and install `qiskit-aer-gpu` instead.


**************************
Tutorials
**************************

A tutorial notebook is available at `here <https://github.com/SotaYoshida/nuqulib/blob/main/tutorial_NuQuLib.ipynb>`_.


*************************
API Reference
*************************

Please refer to :doc:`nuqulib` for detailed API reference.

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
