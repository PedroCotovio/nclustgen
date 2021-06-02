
Nclustgen
---------

Nclustgen is a python tool to generate biclustering and triclustering datasets programmatically.

It wraps two java packages `G-Bic <https://github.com/jplobo1313/G-Bic>`_, and
`G-Tric <https://github.com/jplobo1313/G-Bic>`_, that serve as backend generators. If you are interested on GUI version
of this generator or on using this generator in a java environment checkout those packages!

This tool adds some functionalities to the original packages, for a more fluid interaction with python libraries, like:

- Conversion to numpy arrays
- Conversion to sparse tensors
- Conversion to `networkX <https://networkx.org/>`_ or `dgl <https://www.dgl.ai/>`_ n-partite graphs

Instalation
-----------

Installs by default dgl build with no cuda support, in case you want to use 
gpu install the correct dgl build, more information at: https://www.dgl.ai/pages/start.html.