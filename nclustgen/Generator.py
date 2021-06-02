import inspect
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
class Generator(metaclass=abc.ABCMeta):

    """
    Abstract class from where dimensional specific subclass should inherit. Should not be called directly.
    This class abstracts dimensionality providing core implemented methods and abstract methods that should be
    implemented for any n-clustering generator.
    """

    def __init__(self, n, dstype='NUMERIC', patterns=None, bktype='UNIFORM', clusterdistribution=None,
                 contiguity=None, plaidcoherency='NO_OVERLAPPING', percofoverlappingclusters=0.0,
                 maxclustsperoverlappedarea=0, maxpercofoverlappingelements=0.0, percofoverlappingrows=1.0,
                 percofoverlappingcolumns=1.0, percofoverlappingcontexts=1.0, percmissingsonbackground=0.0,
                 percmissingsonclusters=0.0, percnoiseonbackground=0.0, percnoiseonclusters=0.0, percnoisedeviation=0.0,
                 percerroesonbackground=0.0, percerrorsonclusters=0.0, percerrorondeviation=0.0, silence=False,
                 seed=None, *args, **kwargs):

        """
        Parameters
        ----------

        n: int, internal
            Determines dimensionality (e.g. Bi/Tri clustering). Should only be used by subclasses.
        dstype: {'NUMERIC', 'SYMBOLIC'}, default 'Numeric'
            Type of Dataset to be generated, numeric or symbolic(categorical).
        patterns: list or array, default [['CONSTANT', 'CONSTANT']]
            Defines the type of patterns that will be hidden in the data.
            Shape: number of patterns, number of dimensions
            Patterns_Set: {CONSTANT, ADDITIVE, MULTIPLICATIVE, ORDER_PRESERVING, NONE}
            Numeric_Patterns_Set: {CONSTANT, ADDITIVE, MULTIPLICATIVE, ORDER_PRESERVING, NONE}
            Symbolic_Patterns_Set: {CONSTANT, ORDER_PRESERVING, NONE}
            2D_Numeric_Patterns_Combinations:
                [['Order Preserving', 'None'],
                ['None', 'Order Preserving'],
                ['Constant', 'Constant'],
                ['None', 'Constant'],
                ['Constant', 'None'],
                ['Additive', 'Additive'],
                ['Constant', 'Additive'],
                ['Additive', 'Constant'],
                ['Multiplicative', 'Multiplicative'],
                ['Constant', 'Multiplicative'],
                ['Multiplicative', 'Constant']]
            2D_Symbolic_Patterns_Combinations:
                [['Order Preserving', 'None'],
                ['None', 'Order Preserving'],
                ['Constant', 'Constant'],
                ['None', 'Constant'],
                ['Constant', 'None']]
            3D_Numeric_Patterns_Combinations:
            [['Order Preserving', 'None', 'None'],
            ['None', 'Order Preserving', 'None'],
            ['None', 'None', 'Order Preserving'],
            ['Constant', 'Constant', 'Constant'],
            ['None', 'Constant', 'Constant'],
            ['Constant', 'Constant', 'None'],
            ['Constant', 'None', 'Constant'],
            ['Constant', 'None', 'None'],
            ['None', 'Constant', 'None'],
            ['None', 'None', 'Constant'],
            ['Additive', 'Additive', 'Additive'],
            ['Additive', 'Additive', 'Constant'],
            ['Constant', 'Additive', 'Additive'],
            ['Additive', 'Constant', 'Additive'],
            ['Additive', 'Constant', 'Constant'],
            ['Constant', 'Additive', 'Constant'],
            ['Constant', 'Constant', 'Additive'],
            ['Multiplicative', 'Multiplicative', 'Multiplicative'],
            ['Multiplicative', 'Multiplicative', 'Constant'],
            ['Constant', 'Multiplicative', 'Multiplicative'],
            ['Multiplicative', 'Constant', 'Multiplicative'],
            ['Multiplicative', 'Constant', 'Constant'],
            ['Constant', 'Multiplicative', 'Constant'],
            ['Constant', 'Constant', 'Multiplicative']]
            3D_Symbolic_Patterns_Combinations:
            [['Order Preserving', 'None', 'None'],
            ['None', 'Order Preserving', 'None'],
            ['None', 'None', 'Order Preserving'],
            ['Constant', 'Constant', 'Constant'],
            ['None', 'Constant', 'Constant'],
            ['Constant', 'Constant', 'None'],
            ['Constant', 'None', 'Constant'],
            ['Constant', 'None', 'None'],
            ['None', 'Constant', 'None'],
            ['None', 'None', 'Constant']]
        bktype: {'NORMAL', 'UNIFORM', 'DISCRETE', 'MISSING'}, default 'UNIFORM'
            Determines the distribution used to generate the background values.
        clusterdistribution: list or array, default [['UNIFORM', 4.0, 4.0], ['UNIFORM', 4.0, 4.0]]
            Distribution used to calculate the size of a cluster.
            Shape: number of dimensions, 3 (distribution parameters) -> param1(str), param2(float), param3(float)
                The first parameter(param1) is always the type of distribution {'NORMAL', 'UNIFORM'}.
                If param1==UNIFORM, then param2 and param3 represents the min and max, respectively.
                If param1==NORMAL, then param2 and param3 represents the mean and standard deviation, respectively.
        contiguity: {'COLUMNS', 'CONTEXTS', 'NONE'}, default None
            Contiguity can occur on COLUMNS or CONTEXTS. To avoid contiguity use None.
            If dimensionality == 2 and contiguity == 'CONTEXTS' it defaults to None.
        plaidcoherency: {'ADDITIVE', 'MULTIPLICATIVE', 'INTERPOLED', 'NONE', 'NO_OVERLAPPING'}, default 'NO_OVERLAPPING'
            Enforces the type of plaid coherency. To avoid plaid coherency use NONE, to avoid any overlapping use
            'NO_OVERLAPPING'.
        percofoverlappingclusters: float, default 0.0
            Percentage of overlapping clusters. Defines how many clusters are allowed to overlap.
            Not used if plaidcoherency == 'NO_OVERLAPPING'.
            Range: [0,1]
        maxclustsperoverlappedarea: int, default 0
            Maximum number of clusters overlapped per area. Maximum number of clusters that can overlap together.
            Not used if plaidcoherency == 'NO_OVERLAPPING'.
            Range: [0, nclusters]
        maxpercofoverlappingelements: float, default 0.0
            Maximum percentage of values shared by overlapped clusters.
            Not used if plaidcoherency == 'NO_OVERLAPPING'.
            Range: [0,1]
        percofoverlappingrows: float, default 1.0
            Percentage of allowed amount of overlaping across clusters rows.
            Not used if plaidcoherency == 'NO_OVERLAPPING'.
            Range: [0,1]
        percofoverlappingcolumns: float, default 1.0
            Percentage of allowed amount of overlaping across clusters columns.
            Not used if plaidcoherency == 'NO_OVERLAPPING'.
            Range: [0,1]
        percofoverlappingcontexts: float, default 1.0
            Percentage of allowed amount of overlaping across clusters contexts.
            Not used if plaidcoherency == 'NO_OVERLAPPING' or n >= 3.
            Range: [0,1]
        percmissingsonbackground: float, 0.0
            Percentage of missing values on the background, that is, values that do not belong to planted clusters.
            Range: [0,1]
        percmissingsonclusters: float, 0.0
            Maximum percentage of missing values on each cluster.
            Range: [0,1]
        percnoiseonbackground: float, 0.0
            Percentage of noisy values on background, that is, values with added noise.
            Range: [0,1]
        percnoiseonclusters: float, 0.0
            Maximum percentage of noisy values on each cluster.
            Range: [0,1]
        percnoisedeviation: int or float, 0.0
            Percentage of symbol on noisy values deviation, that is, the maximum difference between the current symbol
            on the matrix and the one that will replaced it to be considered noise.
            If dstype == Numeric then percnoisedeviation -> float else int.
            Ex: Let Alphabet = [1,2,3,4,5] and CurrentSymbol = 3, if the noiseDeviation is '1', then CurrentSymbol will
                be, randomly, replaced by either '2' or '4'. If noiseDeviation is '2', CurrentSymbol can be replaced by
                either '1','2','4' or '5'.
        percerroesonbackground: float, 0.0
            Percentage of error values on background. Similar as noise, a new value is considered an error if the
            difference between it and the current value in the matrix is greater than noiseDeviation.
            Ex: Alphabet = [1,2,3,4,5], If currentValue = 2, and errorDeviation = 2, to turn currentValue an error,
                it's value must be replaced by '5', that is the only possible value that respects
                abs(currentValue - newValue) > noiseDeviation
            Range: [0,1]
        percerrorsonclusters: float, 0.0
            Percentage of errors values on background. Similar as noise, a new value is considered an error if the
            difference between it and the current value in the matrix is greater than noiseDeviation.
            Ex: Alphabet = [1,2,3,4,5], If currentValue = 2, and errorDeviation = 2, to turn currentValue an error,
                it's value must be replaced by '5', that is the only possible value that respects
                abs(currentValue - newValue) > noiseDeviation
            Range: [0,1]
        percerrorondeviation: int or float, 0.0
            Percentage of symbol on error values deviation, that is, the maximum difference between the current symbol
            on the matrix and the one that will replaced it to be considered error.
             If dstype == Numeric then percnoisedeviation -> float else int.
        silence: bool, default False
            If True them the class does not print to the console.
        seed: int, default -1
            Seed to initialize random objects.
            If seed is None or -1 then random objects are initialized without a seed.
        timeprofile: {'RANDOM', 'MONONICALLY_INCREASING', 'MONONICALLY_DECREASING', None}, default None
            It determines a time profile for the ORDER_PRESERVING pattern. Only used if ORDER_PRESERVING in patterns.
            If None and ORDER_PRESERVING in patterns it defaults to 'RANDOM'.
        realval: bool, default True
            Indicates if the dataset is real valued. Only used when dstype == 'NUMERIC'.
        minval: int or float, default -10.0
            Dataset's minimum value. Only used when dstype == 'NUMERIC'.
        maxval: int or float, default 10.0
            Dataset's maximum value. Only used when dstype == 'NUMERIC'.
        symbols: list or array of strings, default None
            Dataset's alphabet (list of possible values/symbols it can contain). Only used if dstype == 'SYMBOLIC'.
            Shape: alphabets length
        nsymbols: int, default 10
            Defines the length of the alphabet, instead of defining specific symbols this parameter can be passed, and
            a list of strings will be create with range(1, n), where n represents this parameter.
            Only used if dstype == 'SYMBOLIC' and symbols is None.
        mean: int or float, default 14.0
            Mean for the background's distribution. Only used when bktype == 'NORMAL'.
        stdev: int or float, default 7.0
            Standard deviation for the background's distribution. Only used when bktype == 'NORMAL'.
        probs: list or array of floats
            Background weighted distribution probabilities. Only used when bktype == 'DISCRETE'.
            No default probabilities, if probs is None and bktype == 'DISCRETE', bktype defaults to 'UNIFORM'.
            Shape: Number of symbols or possible integers
            Range: [0,1]
            Math: sum(probs) == 1
        in_memory: bool, default None
            Determines if generated datasets return dense or sparse matrix (True/False).
            If None then if the generated dataset's size is larger then 10**5 it defaults to sparse, else outputs dense.
            This parameter can be overwritten in the generate method.
        """

        # define dimensions
        self._n = n

        if patterns is None:
            patterns = [['CONSTANT'] * n]
        if clusterdistribution is None:
            clusterdistribution = [['UNIFORM', 4.0, 4.0]] * n
        if seed is None:
            seed = -1

        # Parse basic Parameters
        self.dstype = str(dstype).upper()
        self.patterns = [[str(pattern_type).upper() for pattern_type in pattern] for pattern in patterns]
        self.clusterdistribution = [[str(dist[0]).upper(), float(dist[1]), float(dist[2])] for dist in clusterdistribution]
        self.contiguity = str(contiguity).upper()

        self.time_profile = kwargs.get('timeprofile')
        self.seed = int(seed)

        if self.time_profile:
            self.time_profile = str(self.time_profile).upper()

        # Parse dataset type parameters

        if self.dstype == 'NUMERIC':

            self.realval = bool(kwargs.get('realval', True))
            self.minval = float(kwargs.get('minval', -10.0))
            self.maxval = float(kwargs.get('maxval', 10.0))

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
            self.background = [bktype, float(kwargs.get('mean', 14.0)), float(kwargs.get('sdev', 7.0))]

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
        self._stdout = System.out

    def start_silencing(self, silence=None):

        if silence is None:
            silence = self.silenced

        if bool(silence):
            System.setOut(PrintStream('logs'))

    def stop_silencing(self):

        System.setOut(self._stdout)

        try:
            os.remove('logs')
        except FileNotFoundError:
            pass

    def get_params(self):

        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))

        return {a[0]: a[1]
                for a in attributes if not (a[0].startswith('__') and a[0].endswith('__') or a[0].startswith('_'))
                }

    @abc.abstractmethod
    def initialize_seed(self):
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

        keys = keys[:self._n]

        cluster_type = {2: 'bi', 3: 'Tri'}[self._n]
        geninfo_params = {2: [generatedDataset, False], 3: [generatedDataset]}[self._n]

        js = json.loads(
            str(getattr(generatedDataset, 'get{}csInfoJSON'.format(cluster_type.capitalize()))
                (*geninfo_params).getJSONObject('{}clusters'.format(cluster_type)).toString())
        )

        self.Y = [[js[i][key] for key in keys] for i in js.keys()]

        self.stop_silencing()

        return self.X, self.Y

    @staticmethod
    @abc.abstractmethod
    def dense_to_dgl(x, device, n):
        pass

    @staticmethod
    @abc.abstractmethod
    def dense_to_networkx(x, device=None, n=None):
        pass

    def to_graph(self, x=None, framework='networkx', device='cpu', n=0):

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
            self.graph = getattr(self, 'dense_to_{}'.format(framework))(x, device, n)

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

        # initialize random seed
        self.initialize_seed()

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
