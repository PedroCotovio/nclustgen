
Nclustgen
---------

Nclustgen is a python tool to generate biclustering and triclustering datasets programmatically.

It wraps two java packages `G-Bic <https://github.com/jplobo1313/G-Bic>`_, and
`G-Tric <https://github.com/jplobo1313/G-Bic>`_, that serve as backend generators. If you are interested on a GUI version
of this generator or on using this generator in a java environment check out those packages.

This tool adds some functionalities to the original packages, for a more fluid interaction with python libraries, like:

- Conversion to numpy arrays
- Conversion to sparse tensors
- Conversion to `networkX <https://networkx.org/>`_ or `dgl <https://www.dgl.ai/>`_ n-partite graphs

Instalation
-----------

Pip instructions

Nclustgen installs by default the dgl build with no cuda support, in case you want to use gpu you can override this
by installing the correct dgl build, more information at: https://www.dgl.ai/pages/start.html.

Getting started
---------------

Here are the basics, the full documentation is available at: http://nclustgen.readthedocs.org.

