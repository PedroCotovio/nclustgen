
import abc
import json
import numpy as np

import jpype
import jpype.imports
from jpype.types import *


class Generator(metaclass=abc.ABCMeta):

    def __init__(self, n, dstype='NUMERIC', patterns=None, bktype='UNIFORM', clusterdistribution=None,
                 contiguity=None, plaidcoherency='NO_OVERLAPPING', percofoverlappingclusters=0,
                 maxclustsperoverlappedarea=0, maxpercofoverlappingelements=0.0, percofoverlappingrows=1.0,
                 percofoverlappingcolumns=1.0, percofoverlappingcontexts=1.0, percmissingsonbackground=0.0,
                 percmissingsonclusters=0.0, percnoiseonbackground=0.0, percnoiseonclusters=0.0, percnoisedeviation=0.0,
                 percerroesonbackground=0.0, percerrorsonclusters=0.0, *args, **kwargs):

        # Start JVM
        self.start()

        self.n = n

        if patterns is None:
            patterns = [['CONSTANT'] * n]
        if clusterdistribution is None:
            clusterdistribution = [['UNIFORM', 4, 4]] * n

        self.time_profile = kwargs.get('timeprofile')

        if self.time_profile:
            self.time_profile = str(self.time_profile).upper()

        self.dstype = str(dstype).upper()
        self.patterns = [[str(pattern_type).upper() for pattern_type in pattern] for pattern in patterns]
        self.clusterdistribution = [[str(dist[0]).upper(), int(dist[1]), int(dist[2])] for dist in clusterdistribution]
        self.contiguity = str(contiguity).upper()

        # Dataset type dependent parameters
        self.realval = bool(kwargs.get('realval', True))
        self.minval = int(kwargs.get('minval', -10.0))
        self.maxval = int(kwargs.get('maxval', 10.0))

        try:
            self.symbols = [str(symbol) for symbol in kwargs.get('symbols')]
            self.nsymbols = len(self.symbols)

        except TypeError:
            self.nsymbols = kwargs.get('nsymbols', 10)

            if self.nsymbols:
                self.symbols = [str(symbol) for symbol in range(self.nsymbols)]
            else:
                self.symbols = None

        self.symmetries = kwargs.get('symmetries', False)

        # Overlapping Settings
        self.plaidcoherency = str(plaidcoherency).upper()
        self.percofoverlappingclusts = float(percofoverlappingclusters)
        self.maxclustsperoverlappedarea = int(maxclustsperoverlappedarea)
        self.maxpercofoverlappingelements = float(maxpercofoverlappingelements)
        self.percofoverlappingrows = float(percofoverlappingrows)
        self.percofoverlappingcolumns = float(percofoverlappingcolumns)
        self.percofoverlappingcontexts = float(percofoverlappingcontexts)
        # Noise settings
        self.missing = (float(percmissingsonbackground), float(percmissingsonclusters))
        self.noise = (float(percnoiseonbackground), float(percnoiseonclusters), float(percnoisedeviation))
        self.errors = (float(percerroesonbackground), float(percerrorsonclusters), float(percnoisedeviation))

        # set background
        bktype = str(bktype).upper()
        if bktype == 'NORMAL':
            self.background = [bktype, int(kwargs.get('mean', 14)), kwargs.get('sdev', 7)]

        elif bktype == 'DISCRETE':
            self.background = [bktype, [float(prob) for prob in kwargs.get('probs')]]

        else:
            self.background = [bktype]

        # set bicluster patterns
        self.patterns = [[str(pattern_type).upper() for pattern_type in pattern] for pattern in patterns]

        # dataset
        self.generatedDataset = None
        self.X = None
        self.Y = None
        self.in_memory = kwargs.get('in_memory')

    def get_dstype_vars(self, nrows, ncols, ncontexts, nclusters, background):

        params = [nrows, ncols, ncontexts, nclusters, background]

        if self.dstype == 'NUMERIC':

            params = [self.realval] + params
            params += [self.minval, self.maxval]
            contexts_index = 3
            class_call = 'NumericDatasetGenerator'

        else:

            params += [self.symbols, self.symmetries]
            contexts_index = 2
            class_call = 'SymbolicDatasetGenerator'

        return class_call, params, contexts_index

    @abc.abstractmethod
    def build_background(self):
        pass

    @abc.abstractmethod
    def build_generator(self, class_call, params, contexts_index):
        pass

    @abc.abstractmethod
    def build_patterns(self):
        pass

    @abc.abstractmethod
    def build_structure(self):
        pass

    @abc.abstractmethod
    def build_overlapping(self):
        pass

    def plant_quality_settings(self, generatedDataset):

        generatedDataset.plantMissingElements(*self.missing)
        generatedDataset.plantNoisyElements(*self.noise)
        generatedDataset.plantErrors(*self.errors)

    def asses_memory(self, in_memory=None, **kwargs):

        if in_memory is not None:
            self.in_memory = in_memory
            return in_memory

        elif self.in_memory is not None:
            return self.in_memory

        else:
            gends = kwargs.get('gends')

            try:
                count = gends.getNumRows() * gends.getNumCols() * gends.getNumContexts()

            except AttributeError:
                count = gends.getNumRows() * gends.getNumCols()

            return count < 10**5

    @staticmethod
    @abc.abstractmethod
    def java_to_numpy(generatedDataset, n):
        pass

    @staticmethod
    @abc.abstractmethod
    def java_to_sparse(generatedDataset, n):
        pass

    def to_tensor(self, generatedDataset=None, in_memory=None, keys=None):

        if generatedDataset is None:
            generatedDataset = self.generatedDataset

        if in_memory is None:
            in_memory = self.asses_memory(gends=generatedDataset)

        if keys is None:
            keys = ['X', 'Y', 'Z']

        cluster_type = {2: 'bi', 3: 'Tri'}[self.n]

        # Get Tensor

        if bool(in_memory):
            self.X = self.java_to_numpy(generatedDataset, self.n)

        else:
            self.X = self.java_to_sparse(generatedDataset, self.n)

        # Get clusters

        keys = keys[:self.n]

        js = json.loads(
            str(getattr(generatedDataset, 'get{}csInfoJSON'.format(cluster_type.capitalize()))
                (generatedDataset).getJSONObject('{}clusters'.format(cluster_type)).toString())
        )

        self.Y = [js[i][key] for i in js.keys() for key in keys]

        return self.X, self.Y

    def to_graph(self, x=None, y=None, framework='netx'):

        # TODO implement to graph
        pass

    def generate(self, nrows=100, ncols=100, ncontexts=3, nclusters=1, no_return=False, **kwargs):

        # define background
        background = self.build_background()

        # initialise data generator
        params = self.get_dstype_vars(nrows, ncols, ncontexts, nclusters, background)

        generator = self.build_generator(*params)

        # get patterns
        patterns = self.build_patterns()

        # get structure
        structure = self.build_structure()

        # get overlapping
        overlapping = self.build_overlapping()

        # generate dataset
        generatedDataset = generator.generate(patterns, structure, overlapping)

        # plant missing values, noise & errors
        self.plant_quality_settings(generatedDataset)

        # return
        self.generatedDataset = generatedDataset

        if no_return:
            return None

        return self.to_tensor(generatedDataset, in_memory=kwargs.get('in_memory'))

    @abc.abstractmethod
    def save(self, file_name='example_dataset', path=None, multiple_files=None):
        pass

    @staticmethod
    def start():

        if jpype.isJVMStarted():
            pass
        else:
            # Loading G-Bic
            jpype.startJVM(classpath=['nclustgen/jars/*'])

    @staticmethod
    def shutdown():

        try:
            jpype.shutdownJVM()
        except RuntimeError:
            pass
