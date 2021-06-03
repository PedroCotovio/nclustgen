.. Nclustgen documentation master file, created by
   sphinx-quickstart on Thu Jun  3 12:05:20 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Nclustgen's documentation
=========================

Nclustgen is a python tool to generate biclustering and triclustering datasets programmatically.

It wraps two java packages `G-Bic <https://github.com/jplobo1313/G-Bic>`_, and
`G-Tric <https://github.com/jplobo1313/G-Bic>`_, that serve as backend generators. If you are interested on a GUI version
of this generator or on using this generator in a java environment check out those packages.

This tool adds some functionalities to the original packages, for a more fluid interaction with python libraries, like:

- Conversion to numpy arrays
- Conversion to sparse tensors
- Conversion to `networkX <https://networkx.org/>`_ or `dgl <https://www.dgl.ai/>`_ n-partite graphs

Source Code
^^^^^^^^^^^
The source code is available at: https://github.com/pepedro97/nclustgen.

.. toctree::
   :maxdepth: 4
   :caption: Getting Started

   getting-started/getting_started
   getting-started/generating_data

.. toctree::
   :maxdepth: 4
   :caption: API

   api-reference/generator
   api-reference/bicluster
   api-reference/tricluster



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
