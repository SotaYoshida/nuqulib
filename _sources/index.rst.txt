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

We assume you are using a virtual environment. If not, we recommend creating one.

A recommended environment is `poetry`.

.. code-block:: bash
   :caption: poetry

   poetry install
   poetry run <your_command> <your_args>

If you are using `pip`, we recommend installing the dependencies as follows:

.. code-block:: bash
   :caption: pip

   pip install -r requirements.txt
   pip install -e .



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
