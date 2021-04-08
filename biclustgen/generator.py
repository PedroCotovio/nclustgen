
import jpype
import jpype.imports
from jpype.types import *

# Loading G-Bic
jpype.startJVM(classpath=['biclustgen/jars/G-Bic-1.0.0.jar'])

# Import G-Bic's classes

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

from java.util import ArrayList

# TODO Refactor so that to allow for Heterogeneous and Symbolic data
# TODO All references to dataset type specific operations and variables must run on init (to allow inheritance)


class NumericGenerator:

    def __init__(self, nrows=100, ncols=100, nbics=8, patterns=None, realval=False, minval=0, maxval=1,
                 bktype='UNIFORM', bicsdist=None, contiguity=None, plaidcoherency='NO_OVERLAPPING',
                 percofoverlappingbics=0, maxbicsperoverlappedarea=0, maxpercofoverlappingelements=0.0,
                 percofoverlappingrows=1.0, percofoverlappingcolumns=1.0, percofoverlappingcontexts=1.0, **kwargs):

        if patterns is None:
            patterns = [['CONSTANT', 'CONSTANT']]
        if bicsdist is None:
            bicsdist = [['UNIFORM', 4, 4], ['UNIFORM', 4, 4]]
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

        self.patterns = [[BiclusterType.NUMERIC] + [getattr(PatternType, str(pattern_type).upper())
                                                    for pattern_type in pattern] + [None] for pattern in patterns]

        self.generatedDataset = None

    def generate(self):

        # define background

        background = Background(*self.background)

        # initialise data generator

        generator = NumericDatasetGenerator(self.realval, self.nrows, self.ncols, self.nbics,
                                            background, self.minval, self.maxval)

        # construct patterns

        patterns = ArrayList()
        [patterns.add(SingleBiclusterPattern(*pattern)) for pattern in self.patterns]

        # construct bicluster structure

        structure = BiclusterStructure()
        structure.setRowsSettings(*self.bicsdist[0])
        structure.setColumnsSettings(*self.bicsdist[1])
        structure.setContiguity(self.contiguity)

        # construct bicluster overlapping

        overlapping = OverlappingSettings()
        overlapping.setPlaidCoherency(self.plaidcoherency)
        overlapping.setPercOfOverlappingBics(self.percofoverlappingbics)
        overlapping.setMaxBicsPerOverlappedArea(self.maxbicsperoverlappedarea)
        overlapping.setMaxPercOfOverlappingElements(self.maxpercofoverlappingelements)
        overlapping.setPercOfOverlappingRows(self.percofoverlappingrows)
        overlapping.setPercOfOverlappingColumns(self.percofoverlappingcolumns)

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

        # TODO convert result to numpy array

        return generatedDataset




