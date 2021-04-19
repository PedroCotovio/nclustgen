from Generator import Generator

import numpy as np
from scipy.sparse import csr_matrix

from com.gtric.generator import NumericDatasetGenerator
from com.gtric.service import GtricService
from com.gtric.types import Background
from com.gtric.types import BackgroundType
from com.gtric.types import Contiguity
from com.gtric.types import Distribution
from com.gtric.types import PatternType
from com.gtric.types import BiclusterType
from com.gtric.types import TimeProfile
from com.gtric.types import PlaidCoherency
from com.gtric.utils import InputValidation
from com.gtric.utils import OverlappingSettings
from com.gtric.utils import SingleBiclusterPattern
from com.gtric.utils import BiclusterStructure
from com.gtric.utils import IOUtils as io

from java.util import ArrayList


class TriclusterGenerator(Generator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __to_numpy(self):

        # TODO add biclusters array

        matrix = str(io.matrixToStringColOriented(self.generatedDataset, self.generatedDataset.getNumRows(), 0, False))

        self.matrix = np.array([[int(val) for val in row.split('\t')[1:]] for row in matrix.split('\n')][:-1])

        return self.matrix