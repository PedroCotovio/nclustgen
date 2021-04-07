import jpype
import jpype.imports
from jpype.types import *

jpype.startJVM(classpath=['biclustgen/jars/G-Tric-1.0.2.jar'])

from com.gtric.domain.dataset import NumericDataset
from com.gtric.generator import NumericDatasetGenerator
from com.gtric.generator import TriclusterDatasetGenerator
from com.gtric.service import GTricService
from com.gtric.types import Background
from com.gtric.types import BackgroundType
from com.gtric.types import Contiguity
from com.gtric.types import Distribution
from com.gtric.types import PatternType
from com.gtric.types import PlaidCoherency
from com.gtric.types import TimeProfile
from com.gtric.utils import InputValidation
from com.gtric.utils import OverlappingSettings
from com.gtric.utils import QualitySettings
from com.gtric.utils import TriclusterPattern
from com.gtric.utils import TriclusterStructure

from java.util import ArrayList


class NumericGenerator:

    def __init__(self, nrows=100, ncols=100, nbics=8, patterns=None, realval=False, minval=0, maxval=1,
                 bktype='UNIFORM', bicsdist=None, contiguity=None, plaidcoherency='NO_OVERLAPPING',
                 percofoverlappingbics=0, maxbicsperoverlappedarea=0, maxpercofoverlappingelements=0.0,
                 percofoverlappingrows=1.0, percofoverlappingcolumns=1.0, percofoverlappingcontexts=1.0, **kwargs):

        if patterns is None:
            patterns = [['CONSTANT', 'CONSTANT', 'CONSTANT']]
        if bicsdist is None:
            bicsdist = [['UNIFORM', 4, 4], ['UNIFORM', 4, 4], ['UNIFORM', 4, 4]]
        if contiguity is None:
            contiguity = 'NONE'

        self.nrows = int(nrows)
        self.ncols = int(ncols)
        self.nbics = int(nbics)
        self.realval = bool(realval)
        self.minval = int(minval)
        self.maxval = int(maxval)

        self.bicsdist = [(getattr(Distribution, dist[0]), int(dist[1]), int(dist[2])) for dist in bicsdist]
        self.contiguity = getattr(Contiguity, str(contiguity).upper())

        self.plaidcoherency = getattr(PlaidCoherency, str(plaidcoherency).upper())
        self.percofoverlappingbics = float(percofoverlappingbics)
        self.maxbicsperoverlappedarea = int(maxbicsperoverlappedarea)
        self.maxpercofoverlappingelements = float(maxpercofoverlappingelements)
        self.percofoverlappingrows = float(percofoverlappingrows)
        self.percofoverlappingcolumns = float(percofoverlappingcolumns)
        self.percofoverlappingcontexts = float(percofoverlappingcontexts)

        self.missing = kwargs.get('missing')
        self.noise = kwargs.get('noise')
        self.errors = kwargs.get('errors')

        # set background

        bktype = str(bktype).upper()
        bk = getattr(BackgroundType, bktype)

        if bktype == 'NORMAL':
            self.background = (bk, int(kwargs.get('mean', 14)), kwargs.get('sdev', 7))

        elif bktype == 'DISCRETE':
            self.background = (bk, [float(prob) for prob in kwargs.get('probs')])

        else:
            self.background = tuple([bk])

        # set bicluster patterns

        self.patterns = [[getattr(PatternType, str(pattern_type).upper()) for pattern_type in pattern]
                         for pattern in patterns]

        self.generatedDataset = None

    def generate(self, in_place=False, to_numpy=True):

        # define background

        background = Background(*self.background)

        # initialise data generator

        generator = NumericDatasetGenerator(self.realval, self.nrows, self.nrows, 1, self.nbics,
                                            background, self.minval, self.maxval)

        # construct patterns

        patterns = ArrayList()
        [patterns.add(TriclusterPattern(*pattern)) for pattern in self.patterns]

        # construct bicluster structure

        structure = TriclusterStructure()
        structure.setRowsSettings(*self.bicsdist[0])
        structure.setColumnsSettings(*self.bicsdist[1])
        structure.setContextsSettings(*self.bicsdist[2])
        structure.setContiguity(self.contiguity)

        # construct bicluster overlapping

        overlapping = OverlappingSettings()
        overlapping.setPlaidCoherency(self.plaidcoherency)
        overlapping.setPercOfOverlappingTrics(self.percofoverlappingbics)
        overlapping.setMaxTricsPerOverlappedArea(self.maxbicsperoverlappedarea)
        overlapping.setMaxPercOfOverlappingElements(self.maxpercofoverlappingelements)
        overlapping.setPercOfOverlappingRows(self.percofoverlappingrows)
        overlapping.setPercOfOverlappingColumns(self.percofoverlappingcolumns)
        overlapping.setPercOfOverlappingContexts(self.percofoverlappingcontexts)

        # generate dataset

        generatedDataset = generator.generate(patterns, structure, overlapping)

        # plant missing values, noise & errors

        if self.missing:
            generatedDataset.plantMissingElements(*self.missing)

        if self.noise:
            generatedDataset.plantNoisyElements(*self.noise)

        if self.errors:
            generatedDataset.plantErrors(*self.errors)

        self.generatedDataset = generatedDataset

        if in_place is True:
            self.generatedDataset = generatedDataset

        return generatedDataset




