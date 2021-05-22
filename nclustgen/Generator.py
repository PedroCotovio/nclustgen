
import os
import abc
import warnings
import json
import numpy as np
from sparse import COO

import torch as th

import jpype
import jpype.imports
from jpype.types import *

# Start JVM
if jpype.isJVMStarted():
    pass
else:
    # Loading G-Bic
    jpype.startJVM(classpath=['nclustgen/jars/*'])

from java.lang import System
from java.io import PrintStream


# TODO docs
# TODO add seed
class Generator(metaclass=abc.ABCMeta):

    def __init__(self, n, dstype='NUMERIC', patterns=None, bktype='UNIFORM', clusterdistribution=None,
                 contiguity=None, plaidcoherency='NO_OVERLAPPING', percofoverlappingclusters=0.0,
                 maxclustsperoverlappedarea=0, maxpercofoverlappingelements=0.0, percofoverlappingrows=1.0,
                 percofoverlappingcolumns=1.0, percofoverlappingcontexts=1.0, percmissingsonbackground=0.0,
                 percmissingsonclusters=0.0, percnoiseonbackground=0.0, percnoiseonclusters=0.0, percnoisedeviation=0.0,
                 percerroesonbackground=0.0, percerrorsonclusters=0.0, percerrorondeviation=0.0, silence=False,
                 *args, **kwargs):

        # define dimensions
        self.n = n

        if patterns is None:
            patterns = [['CONSTANT'] * n]
        if clusterdistribution is None:
            clusterdistribution = [['UNIFORM', 4, 4]] * n

        # Parse basic Parameters
        self.dstype = str(dstype).upper()
        self.patterns = [[str(pattern_type).upper() for pattern_type in pattern] for pattern in patterns]
        self.clusterdistribution = [[str(dist[0]).upper(), int(dist[1]), int(dist[2])] for dist in clusterdistribution]
        self.contiguity = str(contiguity).upper()

        self.time_profile = kwargs.get('timeprofile')

        if self.time_profile:
            self.time_profile = str(self.time_profile).upper()

        # Parse dataset type parameters

        if self.dstype == 'NUMERIC':

            self.realval = bool(kwargs.get('realval', True))
            self.minval = int(kwargs.get('minval', -10.0))
            self.maxval = int(kwargs.get('maxval', 10.0))

            # Noise
            self.noise = (float(percnoiseonbackground), float(percnoiseonclusters), float(percnoisedeviation))
            self.errors = (float(percerroesonbackground), float(percerrorsonclusters), float(percerrorondeviation))

        else:
            try:
                self.symbols = [str(symbol) for symbol in kwargs.get('symbols')]
                self.nsymbols = len(self.symbols)

            except TypeError:
                self.nsymbols = kwargs.get('nsymbols', 10)

                if self.nsymbols:
                    self.symbols = [str(symbol) for symbol in range(self.nsymbols)]

            self.symmetries = kwargs.get('symmetries', False)

            # Noise

            self.noise = (float(percnoiseonbackground), float(percnoiseonclusters), int(percnoisedeviation))
            self.errors = (float(percerroesonbackground), float(percerrorsonclusters), int(percerrorondeviation))

        # Overlapping Settings
        self.plaidcoherency = str(plaidcoherency).upper()
        self.percofoverlappingclusts = float(percofoverlappingclusters)
        self.maxclustsperoverlappedarea = int(maxclustsperoverlappedarea)
        self.maxpercofoverlappingelements = float(maxpercofoverlappingelements)
        self.percofoverlappingrows = float(percofoverlappingrows)
        self.percofoverlappingcolumns = float(percofoverlappingcolumns)
        self.percofoverlappingcontexts = float(percofoverlappingcontexts)

        # missing settings
        self.missing = (float(percmissingsonbackground), float(percmissingsonclusters))

        # define background
        bktype = str(bktype).upper()
        if bktype == 'NORMAL':
            self.background = [bktype, int(kwargs.get('mean', 14)), kwargs.get('sdev', 7)]

        elif bktype == 'DISCRETE':

            try:
                self.background = [bktype, [float(prob) for prob in kwargs.get('probs')]]

            except TypeError:
                self.background = ['UNIFORM']

        else:
            self.background = [bktype]

        # initialize class arguments

        # Data
        self.generatedDataset = None
        self.X = None
        self.Y = None
        self.graph = None
        self.in_memory = kwargs.get('in_memory')

        # General
        self.silenced = silence
        self.stdout = System.out

    def start_silencing(self, silence=None):

        if silence is None:
            silence = self.silenced

        if bool(silence):
            System.setOut(PrintStream('logs'))

    def stop_silencing(self):

        System.setOut(self.stdout)

        try:
            os.remove('logs')
        except FileNotFoundError:
            pass

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

    @abc.abstractmethod
    def save(self, file_name='example_dataset', path=None, single_file=None):
        pass

    @staticmethod
    @abc.abstractmethod
    def java_to_numpy(generatedDataset):
        pass

    @staticmethod
    @abc.abstractmethod
    def java_to_sparse(generatedDataset):
        pass

    def to_tensor(self, generatedDataset=None, in_memory=None, keys=None):

        self.start_silencing()

        if generatedDataset is None:
            generatedDataset = self.generatedDataset

        if in_memory is None:
            in_memory = self.asses_memory(gends=generatedDataset)

        if keys is None:
            keys = ['X', 'Y', 'Z']

        # Get Tensor

        if bool(in_memory):
            self.X = self.java_to_numpy(generatedDataset)

        else:
            self.X = self.java_to_sparse(generatedDataset)

        # Get clusters

        keys = keys[:self.n]

        cluster_type = {2: 'bi', 3: 'Tri'}[self.n]
        geninfo_params = {2: [generatedDataset, False], 3: [generatedDataset]}[self.n]

        js = json.loads(
            str(getattr(generatedDataset, 'get{}csInfoJSON'.format(cluster_type.capitalize()))
                (*geninfo_params).getJSONObject('{}clusters'.format(cluster_type)).toString())
        )

        self.Y = [[js[i][key] for key in keys] for i in js.keys()]

        self.stop_silencing()

        return self.X, self.Y

    @staticmethod
    @abc.abstractmethod
    def dense_to_dgl(x, device):
        pass

    @staticmethod
    @abc.abstractmethod
    def dense_to_networkx(x, device=None):
        pass

    def to_graph(self, x=None, framework='networkx', device='cpu'):

        if x is None:
            x = self.X

        # Parse args
        device = str(device).lower()

        if device not in ['cpu', 'gpu']:
            raise AttributeError(
                '{} is not a compatible device, please use either cpu or gpu.'.format(device)
            )

        framework = str(framework).lower()

        if framework not in ['networkx', 'dgl']:
            raise AttributeError(
                '{} is not a compatible framework, please use either dgl or networkx'.format(framework)
            )

        if x is not None:

            # if sparse matrix then transform into dense
            if isinstance(x, COO):
                x = x.todense()

            if device == 'gpu' and not th.cuda.is_available():

                device = 'cpu'

                warnings.warn('CUDA not available CPU will be used instead')

            if device == 'gpu' and framework == 'networkx':

                framework = 'dgl'

                warnings.warn('The Networkx library is not compatible with gpu devices. '
                              'DGL will be used instead.')

            # call private method
            self.graph = getattr(self, 'dense_to_{}'.format(framework))(x, device)

            return self.graph

        else:
            raise AttributeError('No generated dataset exists. '
                                 'Data must first be generated using the .generate() method.')

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

    def asses_memory(self, in_memory=None, **kwargs):

        if in_memory is not None:
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

    def plant_quality_settings(self, generatedDataset):

        generatedDataset.plantMissingElements(*self.missing)
        generatedDataset.plantNoisyElements(*self.noise)
        generatedDataset.plantErrors(*self.errors)

    def generate(self, nrows=100, ncols=100, ncontexts=3, nclusters=1, no_return=False, **kwargs):

        self.start_silencing()

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

        self.stop_silencing()

        if no_return:
            return None, None

        return self.to_tensor(generatedDataset, in_memory=kwargs.get('in_memory'))

    @staticmethod
    def shutdownJVM():

        try:
            jpype.shutdownJVM()
        except RuntimeError:
            pass
