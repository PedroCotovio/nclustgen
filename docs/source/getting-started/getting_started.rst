***************
Getting Started
***************

Installation
------------

This tool can be installed from PyPI:

.. code:: bash

    $ pip install nclustgen

**NOTICE**: Nclustgen installs by default the dgl build with no cuda support, in case you want to use gpu you can override this
by installing the correct dgl build, more information at: https://www.dgl.ai/pages/start.html.

Basic Usage
-----------

Biclustering Dataset
^^^^^^^^^^^^^^^^^^^^

.. seealso:: Detailed API at :doc:`/api-reference/bicluster`.

.. code:: python

    ## Generate biclustering dataset

    from nclustgen import BiclusterGenerator

    # Initialize generator
    generator = BiclusterGenerator(
         dstype='NUMERIC',
         patterns=[['CONSTANT', 'CONSTANT'], ['CONSTANT', 'NONE']],
         bktype='UNIFORM',
         in_memory=True,
         silence=True
    )

    # Get parameters
    generator.params

    # Generate dataset
    x, y = generator.generate(nrows=50, ncols=100, nclusters=3)

    # Build graph
    graph = generator.to_graph(x, framework='dgl', device='cpu')

    # Save data files
    generator.save(file_name='example', single_file=True)

Triclustering Dataset
^^^^^^^^^^^^^^^^^^^^^

.. seealso:: Detailed API at :doc:`/api-reference/tricluster`.

.. code:: python

    ## Generate triclustering dataset

    from nclustgen import TriclusterGenerator

    # Initialize generator
    generator = TriclusterGenerator(
         dstype='NUMERIC',
         patterns=[['CONSTANT', 'CONSTANT', 'CONSTANT'], ['CONSTANT', 'NONE', 'NONE']],
         bktype='UNIFORM',
         in_memory=True,
         silence=True
    )

    # Get parameters
    generator.params

    # Generate dataset
    x, y = generator.generate(nrows=50, ncols=100, ncontexts=10, nclusters=25)

    # Build graph
    graph = generator.to_graph(x, framework='dgl', device='cpu')

    # Save data files
    generator.save(file_name='example', single_file=True)

.. seealso:: This is a basic example, more detail at :doc:`/getting-started/generating_data`.