from .Generator import Generator

import os
import json
import numpy as np
from scipy.sparse import csr_matrix

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

    def to_numpy(self, generatedDataset=None, keys=None):

        if generatedDataset is None:
            generatedDataset = self.generatedDataset

        if keys is None:
            keys = ['X', 'Y', 'Z']

        # Get Tensor

        threshold = int(generatedDataset.getNumRows() / 10)
        steps = [i for i in range(int(generatedDataset.getNumRows() / threshold))]
        tensors = []

        for step in steps:

            tensor = str(io.matrixToStringColOriented(generatedDataset, threshold, step, False))

            tensor = np.array(
                [np.array_split([float(val) for val in row.split('\t')[1:]], self.n) for row in tensor.split('\n')][:-1]
            )

            tensor = tensor.reshape((generatedDataset.getNumContexts(), threshold, generatedDataset.getNumCols()))

            tensors.append(tensor)

        self.X = np.concatenate(tensors, axis=1)

        # Get clusters

        keys = keys[:self.n]

        js = json.loads(
            str(generatedDataset.getTricsInfoJSON(generatedDataset).getJSONObject("Triclusters").toString())
        )

        self.Y = [js[i][key] for i in js.keys() for key in keys]

        return self.X, self.Y

    def to_sparse(self, generatedDataset=None):

        # TODO implement sparse tensor for triclustring

        if generatedDataset is None:
            generatedDataset = self.generatedDataset

    def save(self, file_name='example', path=None, single_file=None):

        serv = GTricService()

        if path is None:
            path = os.getcwd()

        serv.setPath(path)
        serv.setSingleFileOutput(self.asses_memory(single_file, gends=self.generatedDataset))
        serv.saveResult(self.generatedDataset, file_name + 'cluster_data', file_name + 'dataset')