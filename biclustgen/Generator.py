
import numpy as np

# Start JVM

import jpype
import jpype.imports
from jpype.types import *

if jpype.isJVMStarted():
    pass
else:
    # Loading G-Bic
    jpype.startJVM(classpath=['biclustgen/jars/*'])

# TODO Refactor so that to allow for Heterogeneous and Symbolic data
# TODO All references to dataset type specific operations and variables must run on init (to allow inheritance)
# TODO Implement save method

class Generator:

    def __init__(self, dstype='NUMERIC', patterns=None, bktype='UNIFORM', clusterdistribution=None,
                 contiguity=None, plaidcoherency='NO_OVERLAPPING', percofoverlappingclusters=0,
                 maxbicsperoverlappedarea=0, maxpercofoverlappingelements=0.0, percofoverlappingrows=1.0,
                 percofoverlappingcolumns=1.0, percofoverlappingcontexts=1.0, percmissingsonbackground=0.0,
                 percmissingsonclusters=0.0, percnoiseonbackground=0.0, percnoiseonclusters=0.0,
                 percerroesonbackground=0.0, percerrorsonclusters=0.0, *args, **kwargs):

        if contiguity is None:
            contiguity = 'NONE'
        if plaidcoherency is None:
            plaidcoherency = 'NONE'

        self.dstype = str(dstype).upper()
        self.patterns = [[str(pattern_type).upper() for pattern_type in pattern] for pattern in patterns]
        self.clstdistribution = [(str(dist[0]).upper(), int(dist[1]), int(dist[2])) for dist in clusterdistribution]
        self.contiguity = str(contiguity).upper()

        # if self.dstype == 'NUMERIC':
        #
        #     self.realval = bool(kwargs.get('realval', True))
        #     self.minval = int(kwargs.get('minval', -10))
        #     self.maxval = int(kwargs.get('maxval', 10))
        #
        # elif self.dstype == 'SYMBOLIC':
        #
        #     try:
        #         self.symbols = [str(symbol) for symbol in kwargs.get('symbols')]
        #         self.nsymbols = len(self.symbols)
        #
        #     except TypeError:
        #         self.nsymbols = kwargs.get('nsymbols', 10)
        #         self.symbols = [str(symbol) for symbol in range(self.nsymbols)]

        # Overlapping Settings
        self.plaidcoherency = str(plaidcoherency).upper()
        self.percofoverlappingbics = float(percofoverlappingclusters)
        self.maxbicsperoverlappedarea = int(maxbicsperoverlappedarea)
        self.maxpercofoverlappingelements = float(maxpercofoverlappingelements)
        self.percofoverlappingrows = float(percofoverlappingrows)
        self.percofoverlappingcolumns = float(percofoverlappingcolumns)
        self.percofoverlappingcontexts = float(percofoverlappingcontexts)
        # Noise settings
        self.missing = (float(percmissingsonbackground), float(percmissingsonclusters))
        self.noise = (float(percnoiseonbackground), float(percnoiseonclusters))
        self.errors = (float(percerroesonbackground), float(percerrorsonclusters))

        # set background
        bktype = str(bktype).upper()
        if bktype == 'NORMAL':
            self.background = (bktype, int(kwargs.get('mean', 14)), kwargs.get('sdev', 7))

        elif bktype == 'DISCRETE':
            self.background = (bktype, [float(prob) for prob in kwargs.get('probs')])

        else:
            self.background = tuple([bktype])

        # set bicluster patterns
        self.patterns = [[str(pattern_type).upper() for pattern_type in pattern] for pattern in patterns]

        # dataset
        self.generatedDataset = None
        self.X = None
        self.Y = None
        self.in_memory = kwargs.get('in_memory')

    def __background(self):
        pass

    def __generator(self, nrows, ncols, ncontexts, nclusters, background):
        pass

    def __patterns(self):
        pass

    def __structure(self):
        pass

    def __overlapping(self):
        pass

    def __plant_quality_settings(self):
        pass

    def __asses_memory(self, in_memory=None, **kwargs):

        if in_memory is not None:
            self.in_memory = in_memory
            return in_memory

        elif self.in_memory is not None:
            return self.in_memory

        else:
            gends = kwargs.get('gends')

            try:
                count = gends.numRows * gends.numCols * gends.numContexts

            except AttributeError:
                count = gends.numRows * gends.numCols

            return count < 10**5


    def to_numpy(self, generatedDataset):
        pass

    def to_sparse(self, generatedDataset):
        pass

    def generate(self, nrows=100, ncols=100, ncontexts=None, nclusters=1, no_return=False, **kwargs):

        # define background
        background = self.__background()

        # initialise data generator
        generator = self.__generator(nrows, ncols, ncontexts, nclusters, background)

        # get patterns
        patterns = self.__patterns()

        # get structure
        structure = self.__structure()

        # get overlapping
        overlapping = self.__overlapping()

        # generate dataset
        generatedDataset = generator.generate(patterns, structure, overlapping)

        # plant missing values, noise & errors
        self.__plant_quality_settings()

        # return
        self.generatedDataset = generatedDataset

        if no_return:
            return None

        if self.__asses_memory(kwargs.get('in_memory'), gends=generatedDataset):
            self.X, self.Y = self.to_numpy(generatedDataset)

        else:
            self.X, self.Y = self.to_sparse(generatedDataset)

        return self.X, self.Y

    def save(self, file_name='example_dataset', path=None, multiple_files=None):
        pass