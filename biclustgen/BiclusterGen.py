from NumericGenerator import NumericGenerator
from SymbolicGenerator import SymbolicGenerator

import numpy as np
from scipy.sparse import csr_matrix

from com.gbic.generator import NumericDatasetGenerator
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


class BiclusterGenerator(NumericGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __to_numpy(self):

        # TODO add biclusters array

        matrix = str(io.matrixToStringColOriented(self.generatedDataset, self.generatedDataset.getNumRows(), 0, False))

        self.matrix = np.array([[int(val) for val in row.split('\t')[1:]] for row in matrix.split('\n')][:-1])

        return self.matrix