***************
Generating Data
***************

Generators
----------

This tool provides data generators for bi-clustering and tri-clustering, both generators are based on a (dimensions)
abstract class: :doc:`/api-reference/generator`. A complete explanation of the parameters of the generator can be found in
the API reference.

It can generate real-valued, integer, and categorical datasets, with different settings for cluster patterns,
distributions, cluster overlapping, noise, missing values, and other parameters.

.. code:: python

    # Generate real-valued dataset

    from nclustgen import BiclusterGenerator

    # Initialize generator
    generator = BiclusterGenerator(
        # Dataset type
        dstype='NUMERIC',
        # If real-valued
        realval=True,
        minval=1,
        maxval=10
    )

    x, y = generator.generate()
    x

    # Generate categorical dataset

    from nclustgen import BiclusterGenerator

    # Initialize generator
    generator = BiclusterGenerator(
        # Dataset type
        dstype='SYMBOLIC',
        # Number of symbols
        nsymbols=10
    )

    x, y = generator.generate()
    x

A seed argument can also be used to ensure reproducibility:

.. code:: python

    from nclustgen import BiclusterGenerator

    generator = BiclusterGenerator(seed=3)

    x, y = generator.generate()
    x

To generate a dataset, the :meth:`nclustgen.Generator.Generator.generate` method can be called. This method receives as input the dataset's shape and number
of hidden clusters:

.. code:: python

    # Generate bicluster dataset

    from nclustgen import BiclusterGenerator
    generator = BiclusterGenerator()

    x, y = generator.generate(nrows=100, ncols=20, nclusters=20)
    x

    # Generate tricluster dataset

    from nclustgen import TriclusterGenerator
    generator = TriclusterGenerator()

    x, y = generator.generate(nrows=100, ncols=20, ncontexts=3, nclusters=20)
    x

Different patterns can be used for :ref:`biclusters <bic_patterns>` or :ref:`triclusters <tric_patterns>`:

.. code:: python

    # Generate bicluster dataset
    from nclustgen import BiclusterGenerator
    generator = BiclusterGenerator(
        patterns = [['Additive', 'Constant'], ['Constant', 'Multiplicative']]
    )

    x, y = generator.generate()
    x

    # Generate tricluster dataset
    from nclustgen import TriclusterGenerator
    generator = TriclusterGenerator(
        patterns = [['Order_Preserving', 'None', 'None'], ['Constant', 'Constant', 'Constant']]
    )

    x, y = generator.generate()
    x


Biclustering Generator
^^^^^^^^^^^^^^^^^^^^^^

The biclustering generator uses `G-Bic <https://github.com/jplobo1313/G-Bic>`_ a Java library as the backend generator,
check this library if you prefer a graphical interface or to work with Java directly. More information can also be found
there if you wish to modify the actual generator.

.. _bic_patterns:

**Patterns**

The biclustering generator as specified earlier accepts a number of different bicluster patterns here is a complete list:

=========== ====================================
    2D Numeric Patterns Possible Combinations
------------------------------------------------
index       pattern combination
=========== ====================================
0           ['Order Preserving', 'None']
1           ['None', 'Order Preserving']
2           ['Constant', 'Constant']
3           ['None', 'Constant']
4           ['Constant', 'None']
5           ['Additive', 'Additive']
6           ['Constant', 'Additive']
7           ['Additive', 'Constant']
8           ['Multiplicative', 'Multiplicative']
9           ['Constant', 'Multiplicative']
10          ['Multiplicative', 'Constant']
=========== ====================================

=========== ====================================
    2D Symbolic Patterns Possible Combinations
------------------------------------------------
index       pattern combination
=========== ====================================
0           ['Order Preserving', 'None']
1           ['None', 'Order Preserving']
2           ['Constant', 'Constant']
3           ['None', 'Constant']
4           ['Constant', 'None']
=========== ====================================

.. seealso:: Detailed API at :doc:`/api-reference/bicluster`.

Triclustering Generator
^^^^^^^^^^^^^^^^^^^^^^^

The triclustering generator similarly uses `G-Tric <https://github.com/jplobo1313/G-Tric>`_ a Java library as the
backend generator.

.. _tric_patterns:

**Patterns**

Like the biclustering generator, triclustering generator also accepts several different patterns:

=========== ======================================================
        3D Numeric Patterns Possible Combinations
------------------------------------------------------------------
index       pattern combination
=========== ======================================================
0           ['Order Preserving', 'None', 'None']
1           ['None', 'Order Preserving', 'None']
2           ['None', 'None', 'Order Preserving']
3           ['Constant', 'Constant', 'Constant']
4           ['None', 'Constant', 'Constant']
5           ['Constant', 'Constant', 'None']
6           ['Constant', 'None', 'Constant']
7           ['Constant', 'None', 'None']
8           ['None', 'Constant', 'None']
9           ['None', 'None', 'Constant']
10          ['Additive', 'Additive', 'Additive']
11          ['Additive', 'Additive', 'Constant']
12          ['Constant', 'Additive', 'Additive']
13          ['Additive', 'Constant', 'Additive']
14          ['Additive', 'Constant', 'Constant']
15          ['Constant', 'Additive', 'Constant']
16          ['Constant', 'Constant', 'Additive']
17          ['Multiplicative', 'Multiplicative', 'Multiplicative']
18          ['Multiplicative', 'Multiplicative', 'Constant']
19          ['Constant', 'Multiplicative', 'Multiplicative']
20          ['Multiplicative', 'Constant', 'Multiplicative']
21          ['Multiplicative', 'Constant', 'Constant']
22          ['Constant', 'Multiplicative', 'Constant']
23          ['Constant', 'Constant', 'Multiplicative']
=========== ======================================================

=========== ======================================================
        3D Numeric Patterns Possible Combinations
------------------------------------------------------------------
index       pattern combination
=========== ======================================================
0           ['Order Preserving', 'None', 'None']
1           ['None', 'Order Preserving', 'None']
2           ['None', 'None', 'Order Preserving']
3           ['Constant', 'Constant', 'Constant']
4           ['None', 'Constant', 'Constant']
5           ['Constant', 'Constant', 'None']
6           ['Constant', 'None', 'Constant']
7           ['Constant', 'None', 'None']
8           ['None', 'Constant', 'None']
9           ['None', 'None', 'Constant']
=========== ======================================================

.. seealso:: Detailed API at :doc:`/api-reference/tricluster`.

Dense Tensors
-------------

If the generator's *in_memory* parameter is True, then a dense tensor will be generated, in this case
`numpy <https://numpy.org/>`_ is used. If you are not familiar with numpy follow this link to learn more about it:
https://numpy.org/doc/stable/user/quickstart.html

>>> from nclustgen import BiclusterGenerator
>>> generator = BiclusterGenerator(in_memory=True)
>>> x, y = generator.generate()
>>> type(x)
<class 'numpy.ndarray'>

Matrix
^^^^^^

When the generator's output is a dense matrix, it will be of shape *(nrows, ncols)*

>>> from nclustgen import BiclusterGenerator
>>> generator = BiclusterGenerator(in_memory=True)
>>> x, y = generator.generate(nrows=100, ncols=50)
>>> x.shape
(100, 50)

Tensor
^^^^^^

On the other hand, when the generator's output is a dense tensor, it will be of shape *(ncontext, nrows, ncols)*

>>> from nclustgen import TriclusterGenerator
>>> generator = TriclusterGenerator(in_memory=True)
>>> x, y = generator.generate(nrows=100, ncols=50, ncontexts=30)
>>> x.shape
(30, 100, 50)

Sparse Tensors
--------------

If the generator's *in_memory* parameter is False, then a sparse tensor will be generated, in this case different packages
are used depending on the dimensionality of the dataset. But the shape follows the standard set by the dense option.

Matrix
^^^^^^

When the generator's output is a sparse matrix,
`scipy's csr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_ will be used.

>>> from nclustgen import BiclusterGenerator
>>> generator = BiclusterGenerator(in_memory=False)
>>> x, y = generator.generate()
>>> type(x)
<class 'scipy.sparse.csr.csr_matrix'>

Tensor
^^^^^^

On the other hand, when the generator's output is a sparse tensor a
`sparse's COO object <https://sparse.pydata.org/en/stable/construct.html>`_ will be outputted.

>>> from nclustgen import TriclusterGenerator
>>> generator = TriclusterGenerator(in_memory=False)
>>> x, y = generator.generate()
>>> type(x)
<class 'sparse._coo.core.COO'>

Graphs
------

The generator's :meth:`nclustgen.Generator.Generator.to_graph` method allows for either a bipartite or tripartite
graph to be generated, depending on the datasets dimension.

The datasets shape will be transformed in the following way:

**number of nodes** = *nrows + ncols (+ ncontexts)*

**number of edges** = *nrows * ncols (* ncontexts * 3)*

The graphs can be outputted in two different formats as a
`NetworkX Multigraph <https://networkx.org/documentation/stable/tutorial.html#multigraphs>`_, or as a
`DGL heterograph <https://docs.dgl.ai/api/python/dgl.DGLGraph.html>`_ with a
`pytorch <https://pytorch.org/docs/stable/tensors.html>`_ backend.

The `networkX <https://networkx.org/>`_ is a very well known framework to deal with graph data, while
`DGL <https://www.dgl.ai/>`_ is a more recent library mainly for deep learning with graphs, so if you intend to use this
data for deep learning models DGL is recommended, otherwise, networkX will probably be a better option.

>>> from nclustgen import BiclusterGenerator
>>> generator = BiclusterGenerator()
>>> x, y = generator.generate(100, 50)
>>> g = generator.to_graph(framework='dgl')
>>> g
<networkx.classes.graph.Graph object at 0x10a011d60>
>>> len(g.nodes) == 100 + 50
True
>>> len(g.edges) == 100 * 50
True
>>> g = generator.to_graph(framework='dgl')
>>> g
Graph(num_nodes={'col': 50, 'row': 100},
      num_edges={('row', 'elem', 'col'): 5000},
      metagraph=[('row', 'col', 'elem')])
>>> g.num_nodes() == 100 + 50
True
>>> g.num_edges() == 100 * 50
True

In case dgl framework is being used the :meth:`nclustgen.Generator.Generator.to_graph` method can also receive
two additional parameters, the *device* and *cuda* parameters. The first determines if the tensors are stored in cpu or
gpu memory, the second is only used for gpu devices and sets the index of the gpu device to be used in multi-gpu
machines if that's not the case ignore it as it defaults to 0.

>>> g = generator.to_graph(framework='dgl', device='gpu', cuda=0)
>>> g.device
device(type='gpu')