from .Generator import Generator

import os
import json
import numpy as np
from scipy.sparse import csr_matrix

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
from com.gbic.utils import InputValidation
from com.gbic.utils import OverlappingSettings
from com.gbic.utils import SingleBiclusterPattern
from com.gbic.utils import BiclusterStructure
from com.gbic.utils import IOUtils as io

from java.util import ArrayList


class BiclusterGenerator(Generator):

    def __init__(self, *args, **kwargs):
        super().__init__(n=2, *args, **kwargs)

    def build_background(self):

        self.background[0] = getattr(BackgroundType, self.background[0])

        return Background(*self.background)

    def build_generator(self, class_call, params, contexts_index):

        del params[contexts_index]

        return getattr(gen, class_call)(*params)

    def build_patterns(self):

        patterns = ArrayList()

        if self.time_profile:
            self.time_profile = getattr(TimeProfile, self.time_profile)

        [patterns.add(
            SingleBiclusterPattern(
                *[getattr(BiclusterType, self.dstype)] + [getattr(PatternType, pattern_type)
                                                          for pattern_type in pattern] + [self.time_profile]
            )
        ) for pattern in self.patterns]

        return patterns

    def build_structure(self):

        structure = BiclusterStructure()
        structure.setRowsSettings(
            getattr(Distribution, self.clusterdistribution[0][0]), *self.clusterdistribution[0][1:]
        )
        structure.setColumnsSettings(
            getattr(Distribution, self.clusterdistribution[1][0]), *self.clusterdistribution[1][1:]
        )
        structure.setContiguity(getattr(Contiguity, self.contiguity))

        return structure

    def build_overlapping(self):

        overlapping = OverlappingSettings()
        overlapping.setPlaidCoherency(getattr(PlaidCoherency, self.plaidcoherency))
        overlapping.setPercOfOverlappingBics(self.percofoverlappingclusts)
        overlapping.setMaxBicsPerOverlappedArea(self.maxclustsperoverlappedarea)
        overlapping.setMaxPercOfOverlappingElements(self.maxpercofoverlappingelements)
        overlapping.setPercOfOverlappingRows(self.percofoverlappingrows)
        overlapping.setPercOfOverlappingColumns(self.percofoverlappingcolumns)

        return overlapping

    def to_numpy(self, generatedDataset=None, keys=None):

        # TODO break into general to_tensor and specific methods for sparse/numpy bi/triclustring
        # TODO add multiprocessing for chunk processing

        if generatedDataset is None:
            generatedDataset = self.generatedDataset

        if keys is None:
            keys = ['X', 'Y', 'Z']

        # Get Tensor

        matrix = str(io.matrixToStringColOriented(generatedDataset, generatedDataset.getNumRows(), 0, False))

        self.X = np.array([[float(val) for val in row.split('\t')[1:]] for row in matrix.split('\n')][:-1])

        # Get clusters

        keys = keys[:self.n]

        js = json.loads(
            str(generatedDataset.getBicsInfoJSON(generatedDataset, False).getJSONObject("biclusters").toString())
        )

        self.Y = [js[i][key] for i in js.keys() for key in keys]

        return self.X, self.Y

    def to_sparse(self, generatedDataset=None):

        # TODO implement sparse tensor

        if generatedDataset is None:
            generatedDataset = self.generatedDataset

    def save(self, file_name='example', path=None, single_file=None):

        serv = GBicService()

        if path is None:
            path = os.getcwd()

        serv.setPath(path)
        serv.setSingleFileOutput(self.asses_memory(single_file, gends=self.generatedDataset))

        getattr(serv, 'save{}Result'.format(self.dstype.capitalize()))(
            self.generatedDataset, file_name + 'cluster_data', file_name + 'dataset'
        )
