
from .Generator import Generator

import os
import json
import sys
import csv
import numpy as np
from sparse import concatenate, COO

# import dgl without backend info
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
sys.stderr = stderr

import torch as th
import networkx as nx

from com.gtric import generator as gen
from com.gtric.service import GTricService
from com.gtric.types import Background
from com.gtric.types import BackgroundType
from com.gtric.types import Contiguity
from com.gtric.types import Distribution
from com.gtric.types import PatternType
from com.gtric.types import TimeProfile
from com.gtric.types import PlaidCoherency
from com.gtric.utils import OverlappingSettings
from com.gtric.utils import TriclusterStructure
from com.gtric.utils import TriclusterPattern
from com.gtric.utils import RandomObject
from com.gtric.utils import IOUtils as io

from java.util import ArrayList

# helper function
from .utils import tensor_value_check as tvc, loader


class TriclusterGenerator(Generator):
    """
        This class provides an implementation for three-dimensional datasets with hidden triclusters.

        **Examples**

        >>> from nclustgen import TriclusterGenerator
        >>> generator = TriclusterGenerator(
        ...     dstype='NUMERIC',
        ...     patterns=[['CONSTANT', 'CONSTANT', 'CONSTANT'], ['CONSTANT', 'NONE', 'NONE']],
        ...     bktype='UNIFORM',
        ...     in_memory=True,
        ...     silence=True
        ... )
        >>> generator.get_params()
        {'X': None, 'Y': None, 'background': ['UNIFORM'], 'clusterdistribution': [['UNIFORM', 4, 4], ['UNIFORM', 4, 4],
        ['UNIFORM', 4, 4]], 'contiguity': 'NONE', 'dstype': 'NUMERIC', 'errors': (0.0, 0.0, 0.0),
        'generatedDataset': None, 'graph': None, 'in_memory': 'True', 'maxclustsperoverlappedarea': 0,
        'maxpercofoverlappingelements': 0.0, 'maxval': 10.0, 'minval': -10.0, 'missing': (0.0, 0.0), 'cuda': 3,
        'noise': (0.0, 0.0, 0.0), 'patterns': [['CONSTANT', 'CONSTANT', 'CONSTANT'], ['CONSTANT', 'NONE', 'NONE']],
        'percofoverlappingclusts': 0.0, 'percofoverlappingcolumns': 1.0, 'percofoverlappingcontexts': 1.0,
        'percofoverlappingrows': 1.0, 'plaidcoherency': 'NO_OVERLAPPING', 'realval': True, 'seed': -1,
        'silenced': False, 'time_profile': None}
        >>> x, y = generator.generate(nrows=50, ncols=100, ncontexts=5, nclusters=3)
        >>> x
        array([[[-1.29, -4.92, -2.49, ..., -9.17, -5.19,  6.66],
                [ 5.41, -3.04, -1.58, ..., -3.44,  1.99,  9.84],
                [-1.88, -9.09,  5.06, ..., -8.96, -8.4 ,  3.56],
                ...,
                [ 8.74,  4.07,  0.6 , ...,  6.73, -1.3 ,  5.  ],
                [ 3.33,  4.66, -5.72, ...,  0.55,  5.82, -3.17],
                [ 2.32, -9.29,  3.95, ...,  3.61,  3.93, -6.76]],
               [[ 4.34,  8.59, -1.96, ...,  0.88,  8.52, -7.85],
                [ 1.87, -8.59, -9.78, ...,  5.33, -7.45,  3.1 ],
                [-6.86, -3.93,  7.73, ...,  3.21,  6.54, -7.13],
                ...,
                [-5.75,  9.91, -4.76, ...,  0.94, -9.2 , -1.32],
                [ 3.11, -8.26, -2.32, ..., -5.08,  5.33,  2.52],
                [-4.18,  7.98,  8.42, ...,  4.21, -0.03, -7.51]],
               [[ 1.22, -5.69, -8.72, ...,  5.78,  8.74,  1.44],
                [ 3.41, -7.45,  7.01, ...,  8.93, -6.01,  0.18],
                [ 3.8 ,  2.92, -1.87, ..., -1.16, -3.31, -3.02],
                ...,
                [ 4.82, -9.82,  0.31, ...,  9.91, -0.45,  7.86],
                [ 7.24,  8.28, -3.13, ...,  9.12, -0.47,  6.16],
                [-6.61, -7.34, -0.56, ...,  1.41, -1.7 ,  6.22]],
               [[ 3.46,  7.85, -8.23, ...,  1.33,  2.82, -4.05],
                [-8.87, -6.42,  2.28, ...,  9.72, -1.75,  5.01],
                [-0.26, -3.25, -9.16, ..., -1.69,  6.96,  4.63],
                ...,
                [-5.36,  2.84, -2.09, ...,  0.33, -2.88,  3.43],
                [ 5.72,  1.11,  2.11, ...,  0.27, -5.95,  3.39],
                [-7.02, -3.85, -5.44, ...,  1.64, -1.24, -2.74]],
               [[-2.39, -9.27, -8.12, ..., -7.86,  7.54,  4.99],
                [ 2.06,  3.84, -2.99, ...,  4.82, -9.29, -9.23],
                [ 0.21, -5.85, -8.45, ...,  4.35, -2.69,  0.34],
                ...,
                [-0.52, -2.59,  7.63, ..., -8.07, -3.51,  2.7 ],
                [ 4.93, -1.55, -0.65, ..., -0.87,  8.53,  9.97],
                [ 8.03,  2.32, -4.76, ..., -2.03, -4.48, -5.56]]])
        >>> y
        [[[8, 16, 17, 35], [36, 55, 69, 88], [0, 2, 3, 4]], [[7, 21, 33, 35], [22, 57, 65, 75], [0, 1, 2, 4]],
        [[9, 19, 23, 27], [12, 19, 59, 72], [1, 2, 3, 4]]]
        >>> graph = generator.to_graph(x, framework='dgl', device='cpu')
        >>> graph
        Graph(num_nodes={'col': 100, 'ctx': 5, 'row': 50},
              num_edges={('col', 'elem', 'ctx'): 500, ('row', 'elem', 'col'): 5000, ('row', 'elem', 'ctx'): 250},
              metagraph=[('col', 'ctx', 'elem'), ('row', 'col', 'elem'), ('row', 'ctx', 'elem')])
        >>> generator.save(file_name='example', single_file=True)
        """

    def __init__(self, *args, **kwargs):
        super().__init__(n=3, *args, **kwargs)

    def _initialize_seed(self):

        RandomObject.initialization(self.seed)

    def _build_background(self):

        try:
            self.background[0] = getattr(BackgroundType, self.background[0])
        except TypeError:
            pass

        return Background(*self.background)

    def _build_generator(self, class_call, params, contexts_index):

        return getattr(gen, class_call)(*params)

    def _build_patterns(self):

        patterns = ArrayList()

        if self.time_profile:
            self.time_profile = getattr(TimeProfile, str(self.time_profile).upper())

        [patterns.add(
            TriclusterPattern(*[getattr(PatternType, pattern_type) for pattern_type in pattern] + [self.time_profile])
        ) for pattern in self.patterns]

        return patterns

    def _build_structure(self):

        structure = TriclusterStructure()
        structure.setRowsSettings(
            getattr(Distribution, self.clusterdistribution[0][0]), *self.clusterdistribution[0][1:]
        )
        structure.setColumnsSettings(
            getattr(Distribution, self.clusterdistribution[1][0]), *self.clusterdistribution[1][1:]
        )
        structure.setContextsSettings(
            getattr(Distribution, self.clusterdistribution[2][0]), *self.clusterdistribution[2][1:]
        )
        structure.setContiguity(getattr(Contiguity, self.contiguity))

        return structure

    def _build_overlapping(self):

        overlapping = OverlappingSettings()
        overlapping.setPlaidCoherency(getattr(PlaidCoherency, self.plaidcoherency))
        overlapping.setPercOfOverlappingTrics(self.percofoverlappingclusts)
        overlapping.setMaxTricsPerOverlappedArea(self.maxclustsperoverlappedarea)
        overlapping.setMaxPercOfOverlappingElements(self.maxpercofoverlappingelements)
        overlapping.setPercOfOverlappingRows(self.percofoverlappingrows)
        overlapping.setPercOfOverlappingColumns(self.percofoverlappingcolumns)
        overlapping.setPercOfOverlappingContexts(self.percofoverlappingcontexts)

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
            Shape: (ncontexts, nrows, ncols)

        """

        tensor = str(io.matrixToStringColOriented(generatedDataset, generatedDataset.getNumRows(), 0, False))

        tensor = np.array(
            [np.array_split([tvc(val) for val in row.split('\t')[1:]], generatedDataset.getNumContexts())
             for row in tensor.split('\n')][:-1]
        )

        return tensor.reshape(
            (generatedDataset.getNumContexts(), generatedDataset.getNumRows(), generatedDataset.getNumCols())
        )

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

        COO tensor
            Generated dataset as COO tensor.

            **Shape**: (ncontexts, nrows, ncols)

        """

        threshold = int(generatedDataset.getNumRows() / 10)
        steps = [i for i in range(int(generatedDataset.getNumRows() / threshold))]
        tensors = []

        for step in steps:
            tensor = str(io.matrixToStringColOriented(generatedDataset, threshold, step, False))

            tensor = COO.from_numpy(np.array(
                [np.array_split([tvc(val) for val in row.split('\t')[1:]], generatedDataset.getNumContexts())
                 for row in tensor.split('\n')][:-1]
            ))

            tensor = tensor.reshape((generatedDataset.getNumContexts(), threshold, generatedDataset.getNumCols()))

            tensors.append(tensor)

        return concatenate(tensors, axis=1)

    @staticmethod
    def _dense_to_dgl(x, device, cuda=0, nclusters=1, clust_init='zeros'):

        """
        Extracts a tripartite dgl graph from a numpy array

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
            numpy array as tripartite dgl graph.

            **Shape**: (nrows + ncols + ncontexts, nrows * ncols * ncontexts * 3)
        """

        # set (u,v)
        clust_init = loader(th, clust_init)

        tensor = th.tensor(
            [[i, j, z, elem] for z, ctx in enumerate(x) for i, row in enumerate(ctx) for j, elem in enumerate(row)]
        ).T

        graph_data = {
            ('row', 'elem', 'col'): (tensor[0].int(), tensor[1].int()),
            ('row', 'elem', 'ctx'): (tensor[0].int(), tensor[2].int()),
            ('col', 'elem', 'ctx'): (tensor[1].int(), tensor[2].int()),
        }

        # create graph
        G = dgl.heterograph(graph_data)

        # set weights
        G.edges[('row', 'elem', 'col')].data['w'] = tensor[3]
        G.edges[('row', 'elem', 'ctx')].data['w'] = tensor[3]
        G.edges[('col', 'elem', 'ctx')].data['w'] = tensor[3]

        # set cluster members
        for n, axis in enumerate(['ctx', 'row', 'col']):
            for i in range(nclusters):
                G.nodes[axis].data[i] = clust_init(x.shape[n], dtype=th.bool)

        if device == 'gpu':
            G = G.to('cuda:{}'.format(cuda))

        return G

    @staticmethod
    def _dense_to_networkx(x, **kwargs):

        """
        Extracts a tripartite networkx graph from numpy array

        Parameters
        ----------

        x: numpy array
            Data array.
        **kwargs: any, default None
            Additional keywords have no effect but might be accepted for compatibility.

        Returns
        -------

        Graph object
            numpy array as tripartite networkx graph.

            **Shape**: (nrows + ncols + ncontexts, nrows * ncols * ncontexts * 3)

        """

        G = nx.MultiGraph()

        edges = np.array(
            [[('row-{}'.format(i), 'col-{}'.format(j), elem),
              ('row-{}'.format(i), 'ctx-{}'.format(z), elem),
              ('col-{}'.format(j), 'ctx-{}'.format(z), elem)]
             for z, ctx in enumerate(x) for i, row in enumerate(ctx) for j, elem in enumerate(row)]
        )

        # reshape from (elements, n, edge) to (edges, edge)
        edges = edges.reshape(edges.shape[0] * edges.shape[1], edges.shape[2])

        G.add_weighted_edges_from(edges)

        return G

    def save(self, extension='default', file_name='example', path=None, single_file=None, **kwargs):

        """
        Saves data files to chosen path.

        Parameters
        ----------

        extension: {'default', 'csv'}, default 'default'
            Extension of saved data file. If default, uses Java class default. Else it returns a data file per context.
        file_name: str, default 'example_dataset'
            Saved files prefix.
        path: str, default None
            Path to save files. If None then files are saved in the current working directory.
        single_file: Bool, default None.
            If False dataset is saved in multiple data files. If None then if the dataset's size is larger then 10**5
            it defaults to False, else True. Only used if extension=='default'.
        **kwargs: any, default None
            Additional keywords that are passed on.

        Examples
        --------

        >>> generator = TriclusterGenerator(silence=True)
        >>> generator.generate()
        >>> generator.save(file_name='TricFiles', single_file=False)
        >>> generator.save(extension='csv', file_name='TricFiles', delimiter=';')

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

            elif isinstance(self.X, COO):
                self.X = self._java_to_numpy(self.generatedDataset)

            # save data
            for i, arr in enumerate(self.X):
                np.savetxt('{}_dataset_ctx{}.csv'.format(os.path.join(path, file_name), i), arr, fmt="%d", **kwargs)

            # save json

            with open('{}_cluster_data.json'.format(os.path.join(path, file_name)), 'w') as outfile:
                json.dump(self.cluster_info, outfile)

            # save txt
            with open('{}_cluster_data.txt'.format(os.path.join(path, file_name)), 'w') as outfile:
                outfile.write(str(self.generatedDataset.getTricsInfo()))

        else:

            serv = GTricService()

            serv.setPath(path)
            serv.setSingleFileOutput(self._asses_memory(single_file, gends=self.generatedDataset))
            serv.saveResult(self.generatedDataset, file_name + '_cluster_data', file_name + '_dataset')

        self._stop_silencing()


class TriclusterGeneratorbyConfig(TriclusterGenerator):

    """
    This class initializes the generator via configuration file.

    **Examples**

    >>> from nclustgen import TriclusterGeneratorbyConfig
    >>> generator = TriclusterGeneratorbyConfig('example.json')
    >>> x, y = generator.generate(nrows=50, ncols=100, ncontexte=4, nclusters=2)
    >>> x
    array([[[ 3.94, -7.62, -2.68, ..., -1.66,  4.41, -3.8 ],
            [-2.27, -7.19, -3.42, ...,  7.19, -2.9 , -6.03],
            [-8.91, -9.46, -7.98, ..., -0.78, -7.66, -4.96],
            ...,
            [-7.93,  9.79,  2.95, ...,  2.01,  7.99,  6.15],
            [-4.25, -3.81, -1.43, ..., -0.61, -5.36, -8.09],
            [ 0.4 , -5.36, -3.68, ...,  8.5 ,  6.8 , -7.34]],
           [[ 0.62, -1.18, -3.07, ...,  0.23, -8.38,  2.96],
            [ 6.37,  4.63,  6.15, ...,  9.13,  9.6 ,  9.5 ],
            [-5.33,  0.15,  1.65, ...,  5.73, -4.64, -6.47],
            ...,
            [ 9.16,  4.75,  3.06, ...,  3.76, -3.09, -6.96],
            [ 3.6 ,  5.54, -0.2 , ...,  1.09,  9.23, -0.62],
            [ 2.68, -6.15, -8.99, ...,  8.65,  9.89,  7.63]],
           [[ 0.55, -1.03,  6.35, ...,  3.88,  5.96, -6.52],
            [-0.71,  7.99,  2.56, ..., -7.15,  0.33,  7.9 ],
            [ 0.86,  2.99,  3.69, ...,  1.57, -5.23,  4.59],
            ...,
            [ 4.2 ,  4.03, -9.11, ...,  5.28,  6.09,  1.19],
            [-0.31,  7.71,  7.57, ..., -3.57, -9.67, -9.89],
            [ 6.55,  4.69, -9.96, ..., -8.9 ,  7.31, -0.13]]])

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
