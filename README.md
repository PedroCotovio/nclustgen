
# Nclustgen

Nclustgen is a python tool to generate biclustering and triclustering datasets programmatically.

It wraps two java packages [G-Bic](https://github.com/jplobo1313/G-Bic), and
[G-Tric](https://github.com/jplobo1313/G-Bic), that serve as backend generators. If you are interested on a GUI version
of this generator or on using this generator in a java environment check out those packages.

This tool adds some functionalities to the original packages, for a more fluid interaction with python libraries, like:

* Conversion to numpy arrays
* Conversion to sparse tensors
* Conversion to [networkX](https://networkx.org/) or [dgl](https://www.dgl.ai/) n-partite graphs

## Installation

This tool can be installed from PyPI:

```sh
pip install nclustgen
```

**NOTICE**: Nclustgen installs by default the dgl build with no cuda support, in case you want to use gpu you can override this
by installing the correct dgl build, more information at: https://www.dgl.ai/pages/start.html.

## Getting started

Here are the basics, the full documentation is available at: http://nclustgen.readthedocs.org.

```python

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
generator.get_params()

# Generate dataset
x, y = generator.generate(nrows=50, ncols=100, nclusters=3)

# Build graph
graph = generator.to_graph(x, framework='dgl', device='cpu')

# Save data files
generator.save(file_name='example', single_file=True)

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
generator.get_params()

# Generate dataset
x, y = generator.generate(nrows=50, ncols=100, ncontexts=10, nclusters=25)

# Build graph
graph = generator.to_graph(x, framework='dgl', device='cpu')

# Save data files
generator.save(file_name='example', single_file=True)
```

## License
[GPLv3](LICENSE)

## Documentation
The documentation is available at: https://nclustgen.readthedocs.org.

