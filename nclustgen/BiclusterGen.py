from .Generator import Generator

import os
import json
import sys
import csv
import numpy as np
from scipy.sparse import csr_matrix, vstack

# import dgl without backend info
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
sys.stderr = stderr

import torch as th
import networkx as nx

from com.gbic import generator as gen
from com.gbic.service import GBicService
from com.gbic.types import Background
from com.gbic.types import BackgroundType
from com.gbic.types import Contiguity
from com.gbic.types import Distribution
from com.gbic.types import PatternType
from com.gbic.types import BiclusterType
from com.gbic.types import TimeProfile
from com.gbic.types import PlaidCoherency
from com.gbic.utils import OverlappingSettings
from com.gbic.utils import SingleBiclusterPattern
from com.gbic.utils import BiclusterStructure
from com.gbic.utils import RandomObject
from com.gbic.utils import IOUtils as io

from java.util import ArrayList

# helper function
from .utils import tensor_value_check as tvc, loader


class BiclusterGenerator(Generator):

    """
    This class provides an implementation for two-dimensional datasets with hidden biclusters.

    **Examples**

    >>> from nclustgen import BiclusterGenerator
    >>> generator = BiclusterGenerator(
    ...     dstype='NUMERIC',
    ...     patterns=[['CONSTANT', 'CONSTANT'], ['CONSTANT', 'NONE']],
    ...     bktype='UNIFORM',
    ...     in_memory=True,
    ...     silence=True
    ... )
    >>> generator.get_params()
    {'X': None, 'Y': None, 'background': ['UNIFORM'], 'clusterdistribution': [['UNIFORM', 4, 4], ['UNIFORM', 4, 4]],
    'contiguity': 'NONE', 'dstype': 'NUMERIC', 'errors': (0.0, 0.0, 0.0), 'generatedDataset': None, 'graph': None,
    'in_memory': 'True', 'maxclustsperoverlappedarea': 0, 'maxpercofoverlappingelements': 0.0, 'maxval': 10.0,
    'minval': -10.0, 'missing': (0.0, 0.0), 'cuda': 2, 'noise': (0.0, 0.0, 0.0),
    'patterns': [['CONSTANT', 'CONSTANT'], ['CONSTANT', 'NONE']], 'percofoverlappingclusts': 0.0,
    'percofoverlappingcolumns': 1.0, 'percofoverlappingcontexts': 1.0, 'percofoverlappingrows': 1.0,
    'plaidcoherency': 'NO_OVERLAPPING', 'realval': True, 'seed': -1, 'silenced': True, 'time_profile': None}
    >>> x, y = generator.generate(nrows=50, ncols=100, nclusters=3)
    >>> x
    array([[-4.43, -8.2 , -0.34, ...,  8.85,  9.24,  6.13],
           [ 9.28,  9.45,  5.46, ...,  7.83,  8.67, -6.48],
           [-9.97, -2.14, -6.58, ...,  1.23,  5.64, -7.29],
           ...,
           [-5.12,  1.11, -3.44, ..., -7.45, -0.21,  2.21],
           [-0.96,  5.43, -3.28, ...,  9.58, -0.73,  3.99],
           [-0.75,  8.91, -6.91, ..., -9.22,  0.43, -4.46]])
    >>> y
    [[[1, 9, 37, 46], [13, 25, 32, 79]], [[17, 29, 39, 46], [0, 5, 74, 90]], [[21, 30, 39, 42], [8, 46, 60, 93]]]
    >>> graph = generator.to_graph(x, framework='dgl', device='cpu')
    >>> graph
    Graph(num_nodes={'col': 100, 'row': 50},
          num_edges={('row', 'elem', 'col'): 5000},
          metagraph=[('row', 'col', 'elem')])
    >>> generator.save(file_name='example', single_file=True)

    """

    def __init__(self, *args, **kwargs):
        super().__init__(n=2, *args, **kwargs)

    def _initialize_seed(self):

        RandomObject.initialization(self.seed)

    def _build_background(self):

        try:
            self.background[0] = getattr(BackgroundType, self.background[0])
        except TypeError:
            pass

        return Background(*self.background)

    def _build_generator(self, class_call, params, contexts_index):

        del params[contexts_index]

        return getattr(gen, class_call)(*params)

    def _build_patterns(self):

        patterns = ArrayList()

        if self.time_profile:
            self.time_profile = getattr(TimeProfile, str(self.time_profile).upper())

        [patterns.add(
            SingleBiclusterPattern(
                *[getattr(BiclusterType, self.dstype)] + [getattr(PatternType, pattern_type)
                                                          for pattern_type in pattern] + [self.time_profile]
            )
        ) for pattern in self.patterns]

        return patterns

    def _build_structure(self):

        structure = BiclusterStructure()
        structure.setRowsSettings(
            getattr(Distribution, self.clusterdistribution[0][0]), *self.clusterdistribution[0][1:]
        )
        structure.setColumnsSettings(
            getattr(Distribution, self.clusterdistribution[1][0]), *self.clusterdistribution[1][1:]
        )
        if self.contiguity == 'CONTEXTS':

            self.contiguity = 'NONE'

        structure.setContiguity(getattr(Contiguity, self.contiguity))

        return structure

    def _build_overlapping(self):

        overlapping = OverlappingSettings()
        overlapping.setPlaidCoherency(getattr(PlaidCoherency, self.plaidcoherency))
        overlapping.setPercOfOverlappingBics(self.percofoverlappingclusts)
        overlapping.setMaxBicsPerOverlappedArea(self.maxclustsperoverlappedarea)
        overlapping.setMaxPercOfOverlappingElements(self.maxpercofoverlappingelements)
        overlapping.setPercOfOverlappingRows(self.percofoverlappingrows)
        overlapping.setPercOfOverlappingColumns(self.percofoverlappingcolumns)

        return overlapping

    @staticmethod
    def _java_to_numpy(generatedDataset):

        """
        Extracts numpy array from Dataset object.

        Parameters
        ----------

        generatedDataset: Dataset object
            Generated dataset (java object).

        Returns
        -------

        numpy array
            Generated dataset as numpy array.
            Shape: (nrows, ncols)

        """

        tensor = str(io.matrixToStringColOriented(generatedDataset, generatedDataset.getNumRows(), 0, False))

        return np.array([[tvc(val) for val in row.split('\t')[1:]] for row in tensor.split('\n')][:-1])

    @staticmethod
    def _java_to_sparse(generatedDataset):

        """
        Extracts sparce tensor from Dataset object.

        Parameters
        ----------

        generatedDataset: Dataset object
            Generated dataset (java object).

        Returns
        -------

        csr_matrix
            Generated dataset as csr_matrix.

            **Shape**: (nrows, ncols)
        """

        threshold = int(generatedDataset.getNumRows() / 10)
        steps = [i for i in range(int(generatedDataset.getNumRows() / threshold))]
        tensors = []

        for step in steps:
            tensor = str(io.matrixToStringColOriented(generatedDataset, threshold, step, False))

            tensor = csr_matrix([[tvc(val) for val in row.split('\t')[1:]] for row in tensor.split('\n')][:-1])

            tensors.append(tensor)

        return vstack(tensors)

    @staticmethod
    def _dense_to_dgl(x, device, cuda=0, nclusters=1, clust_init='zeros'):

        """
        Extracts a bipartite dgl graph from a numpy array

        Parameters
        ----------

        x: numpy array
            Data array.
        device: {'cpu', 'gpu'}
            Type of device for storing the tensor.
        cuda: int, default 0
            Index of cuda device to use. Only used if device==True.
        nclusters: int, default 1
            Number of clusters to be initialized in graph.
        clust_init: str or function, default 'zeros'
            Function to initialize clusters. If string it should be a function available in torch. Else it should point
            to a function with inputs in form (shape, dtype).
        Returns
        -------

        heterograph object
            numpy array as bipartite dgl graph.

            **Shape**: (nrows + ncols, nrows * ncols)

        """

        # set (u,v)
        clust_init = loader(th, clust_init)

        tensor = th.tensor([[i, j, elem] for i, row in enumerate(x) for j, elem in enumerate(row)]).T

        graph_data = {
               ('row', 'elem', 'col'): (tensor[0].int(), tensor[1].int()),
            }

        # create graph
        G = dgl.heterograph(graph_data)

        # set weights
        weights = tensor[2]

        G.edata['w'] = weights

        # set cluster members

        for n, axis in enumerate(['row', 'col']):
            for i in range(nclusters):
                G.nodes[axis].data[i] = clust_init(x.shape[n], dtype=th.bool)

        if device == 'gpu':
            G = G.to('cuda:{}'.format(cuda))

        return G

    @staticmethod
    def _dense_to_networkx(x, **kwargs):

        """
        Extracts a bipartite networkx graph from numpy array

        Parameters
        ----------

        x: numpy array
            Data array.
        **kwargs: any, default None
            Additional keywords have no effect but might be accepted for compatibility.

        Returns
        -------

        Graph object
            numpy array as bipartite networkx graph.

            **Shape**: (nrows + ncols, nrows * ncols)

        """

        G = nx.Graph()

        for n, axis in enumerate(['row', 'col']):

            G.add_nodes_from(
                (('{}-{}'.format(axis, i), {'cluster': 0}) for i in range(x.shape[n])), bipartite=n)

        G.add_weighted_edges_from(
            [('row-{}'.format(i), 'col-{}'.format(j), elem)
             for i, row in enumerate(x) for j, elem in enumerate(row)]
        )

        return G

    def save(self, extension='default', file_name='example', path=None, single_file=None, **kwargs):

        """
        Saves data files to chosen path.

        Parameters
        ----------

        extension: {'default', 'csv'}, default 'default'
            Extension of saved data file. If default, uses Java class default.
        file_name: str, default 'example_dataset'
            Saved files prefix.
        path: str, default None
            Path to save files. If None then files are saved in the current working directory.
        single_file: Bool, default None.
            If False dataset is saved in multiple data files. If None then if the dataset's size is larger then 10**5
            it defaults to False, else True.
        **kwargs: any, default None
            Additional keywords that are passed on.

        Examples
        --------

        >>> generator = BiclusterGenerator(silence=True)
        >>> generator.generate()
        >>> generator.save(file_name='BicFiles', single_file=False)
        >>> generator.save(extension='csv', file_name='BicFiles', delimiter=';')

        """

        if path is None:
            path = os.getcwd() + '/'

        self._start_silencing()

        if extension == 'csv':
            # check if dense exists
            if self.generatedDataset is None:
                raise AttributeError('No generated dataset exists. '
                                     'Data must first be generated using the .generate() method.')

            elif self.X is None:
                _, _ = self.to_tensor(in_memory=False)

            elif isinstance(self.X, csr_matrix):
                self.X = self._java_to_numpy(self.generatedDataset)

            # save data

            if not self._asses_memory(single_file, gends=self.generatedDataset):

                for i, array in enumerate(np.split(self.X, 10)):
                    np.savetxt('{}_dataset_{}.csv'.format(os.path.join(path, file_name), i), array, fmt="%d", **kwargs)

            else:
                np.savetxt('{}_dataset.csv'.format(os.path.join(path, file_name)), self.X, fmt="%d", **kwargs)

            # save json

            with open('{}_cluster_data.json'.format(os.path.join(path, file_name)), 'w') as outfile:
                json.dump(self.cluster_info, outfile)

            # save txt
            with open('{}_cluster_data.txt'.format(os.path.join(path, file_name)), 'w') as outfile:
                outfile.write(str(self.generatedDataset.getBicsInfo()))

        else:

            serv = GBicService()

            serv.setPath(path)
            serv.setSingleFileOutput(self._asses_memory(single_file, gends=self.generatedDataset))

            getattr(serv, 'save{}Result'.format(self.dstype.capitalize()))(
                self.generatedDataset, file_name + '_cluster_data', file_name + '_dataset'
            )

        self._stop_silencing()


class BiclusterGeneratorbyConfig(BiclusterGenerator):

    """
    This class initializes the generator via configuration file.

    **Examples**

    >>> from nclustgen import BiclusterGeneratorbyConfig
    >>> generator = BiclusterGeneratorbyConfig('example.json')
    >>> generator.get_params()
    {'X': None, 'Y': None, 'background': ['UNIFORM'], 'clusterdistribution': [['UNIFORM', 4, 4], ['UNIFORM', 4, 4]],
    'contiguity': 'NONE', 'dstype': 'NUMERIC', 'errors': (0.0, 0.0, 0.0), 'generatedDataset': None, 'graph': None,
    'in_memory': 'True', 'maxclustsperoverlappedarea': 0, 'maxpercofoverlappingelements': 0.0, 'maxval': 10.0,
    'minval': -10.0, 'missing': (0.0, 0.0), 'noise': (0.0, 0.0, 0.0),
    'patterns': [['CONSTANT', 'CONSTANT'], ['CONSTANT', 'NONE']], 'percofoverlappingclusts': 0.0,
    'percofoverlappingcolumns': 1.0, 'percofoverlappingcontexts': 1.0, 'percofoverlappingrows': 1.0,
    'plaidcoherency': 'NO_OVERLAPPING', 'realval': True, 'seed': -1, 'silenced': False, 'time_profile': None}
    >>> x, y = generator.generate(nrows=50, ncols=100, nclusters=3)
    >>> x
    array([[-4.67,  3.57,  2.38, ..., -7.41, -4.14,  4.64],
           [-8.31,  8.06,  1.33, ..., -7.24, -2.62, -5.59],
           [-4.68, -5.43, -1.81, ..., -0.49, -1.34,  0.68],
           ...,
           [ 1.85,  9.55,  8.1 , ..., -2.5 ,  2.41, -5.54],
           [-2.09,  0.73,  6.38, ...,  0.46, -8.97,  4.46],
           [-7.21,  6.6 , -9.78, ..., -6.29, -7.24, -2.98]])

    """

    def __init__(self, file_path=None):

        """
        **Parameters**

        file_path: str, default None
            Determines the path to the configuration file. If None then no parameters are passed to class.
        """
        if file_path:
            f = open(file_path, )
            params = json.load(f)
            f.close()

            super().__init__(**params)

        else:
            super().__init__()
