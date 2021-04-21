from Generator import Generator

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
        super().__init__(*args, **kwargs)

        if len(self.patterns[0]) == 1:
            self.patterns = [pattern * 2 for pattern in self.patterns]

        if len(self.clusterdistribution) == 1:
            self.clusterdistribution = self.clusterdistribution * 2

    def __background(self):

        self.background[0] = getattr(BackgroundType, self.background[0])

        return Background(*self.background)

    def __generator(self, class_call, params, contexts_index):

        del params[contexts_index]

        return getattr(gen, class_call)(*params)

    def __patterns(self):

        patterns = ArrayList()

        [patterns.add(SingleBiclusterPattern([getattr(BiclusterType, self.dstype)] + [getattr(PatternType, pattern_type)
                                                                                      for pattern_type in pattern] +
                                             [self.time_profile])) for pattern in self.patterns]

    def __structure(self):

        structure = BiclusterStructure()
        structure.setRowsSettings(*self.clusterdistribution[0])
        structure.setColumnsSettings(*self.clusterdistribution[1])
        structure.setContiguity(self.contiguity)

        return structure

    def __overlapping(self):

        overlapping = OverlappingSettings()
        overlapping.setPlaidCoherency(self.plaidcoherency)
        overlapping.setPercOfOverlappingBics(self.percofoverlappingbics)
        overlapping.setMaxBicsPerOverlappedArea(self.maxbicsperoverlappedarea)
        overlapping.setMaxPercOfOverlappingElements(self.maxpercofoverlappingelements)
        overlapping.setPercOfOverlappingRows(self.percofoverlappingrows)
        overlapping.setPercOfOverlappingColumns(self.percofoverlappingcolumns)

        return overlapping

    def to_numpy(self, generatedDataset):

        # TODO add biclusters array
        # TODO break into general to_matrix and specific methods for sparse or numpy
        # TODO add multiprocessing for chunk processing

        matrix = str(io.matrixToStringColOriented(self.generatedDataset, self.generatedDataset.getNumRows(), 0, False))

        self.matrix = np.array([[int(val) for val in row.split('\t')[1:]] for row in matrix.split('\n')][:-1])

        return self.matrix

    def to_sparse(self, generatedDataset):

        # TODO implement
        pass

    def save(self, file_name='example_dataset', path=None, multiple_files=None):

        # TODO implement by calling from G-bic
        pass
