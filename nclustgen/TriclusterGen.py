
from .Generator import Generator

import os
import json
import sys
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
from com.gtric.utils import IOUtils as io

from java.util import ArrayList


class TriclusterGenerator(Generator):

    def __init__(self, *args, **kwargs):
        super().__init__(n=3, *args, **kwargs)

    def build_background(self):

        self.background[0] = getattr(BackgroundType, self.background[0])

        return Background(*self.background)

    def build_generator(self, class_call, params, contexts_index):

        return getattr(gen, class_call)(*params)

    def build_patterns(self):

        patterns = ArrayList()

        [patterns.add(
            TriclusterPattern(*[getattr(PatternType, pattern_type) for pattern_type in pattern])
        ) for pattern in self.patterns]

        # TODO add set_timeprofile

        return patterns

    def build_structure(self):

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

    def build_overlapping(self):

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
    def java_to_numpy(generatedDataset):

        tensor = str(io.matrixToStringColOriented(generatedDataset, generatedDataset.getNumRows(), 0, False))

        tensor = np.array(
            [np.array_split([float(val) for val in row.split('\t')[1:]], 3) for row in tensor.split('\n')][:-1]
        )

        return tensor.reshape(
            (generatedDataset.getNumContexts(), generatedDataset.getNumRows(), generatedDataset.getNumCols())
        )

    @staticmethod
    def java_to_sparse(generatedDataset):

        threshold = int(generatedDataset.getNumRows() / 10)
        steps = [i for i in range(int(generatedDataset.getNumRows() / threshold))]
        tensors = []

        for step in steps:
            tensor = str(io.matrixToStringColOriented(generatedDataset, threshold, step, False))

            tensor = COO.from_numpy(np.array(
                [np.array_split([float(val) for val in row.split('\t')[1:]], 3) for row in tensor.split('\n')][:-1]
            ))

            tensor = tensor.reshape((generatedDataset.getNumContexts(), threshold, generatedDataset.getNumCols()))

            tensors.append(tensor)

        return concatenate(tensors, axis=1)

    @staticmethod
    def dense_to_dgl(x, device):

        # TODO implement dense_to_dgl tric
        # graph_data = {
        #    ('row', 'elem', 'col'): (th.tensor([0, 1]), th.tensor([1, 2])),
        # }
        pass

    @staticmethod
    def dense_to_netwrokx(x, device=None):

        G = nx.Graph()

        for n, axis in enumerate(['ctx', 'row', 'col']):

            G.add_nodes_from(
                (('{}-{}'.format(axis, i), {'cluster': 0}) for i in range(x.shape[n])), bipartide=n)

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

    def save(self, file_name='example', path=None, single_file=None):

        serv = GTricService()

        if path is None:
            path = os.getcwd()

        serv.setPath(path)
        serv.setSingleFileOutput(self.asses_memory(single_file, gends=self.generatedDataset))
        serv.saveResult(self.generatedDataset, file_name + 'cluster_data', file_name + 'dataset')


class TriclusterGeneratorbyConfig(TriclusterGenerator):

    def __init__(self, file_path=None):
        if file_path:
            f = open(file_path, )
            params = json.load(f)
            f.close()

            super().__init__(**params)

        super().__init__()
