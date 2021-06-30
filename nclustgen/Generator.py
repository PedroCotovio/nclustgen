import inspect
import os
import abc
import warnings
import json

import numpy as np
from sparse import COO
from scipy.sparse import csr_matrix

import torch as th

import jpype
import jpype.imports

from java.lang import System
from java.io import PrintStream


class Generator(metaclass=abc.ABCMeta):

    """
    Abstract class from where dimensional specific subclasses should inherit. Should not be called directly.
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

            **Shape**: (number of patterns, number of dimensions)

            **Patterns_Set**: {CONSTANT, ADDITIVE, MULTIPLICATIVE, ORDER_PRESERVING, NONE}

            **Numeric_Patterns_Set**: {CONSTANT, ADDITIVE, MULTIPLICATIVE, ORDER_PRESERVING, NONE}

            **Symbolic_Patterns_Set**: {CONSTANT, ORDER_PRESERVING, NONE}

            **Pattern_Combinations**:

                =========== ====================================
                    2D Numeric Patterns Possible Combinations
                ------------------------------------------------
                index       pattern combination
                =========== ====================================
                0           ['Order_Preserving', 'None']
                1           ['None', 'Order_Preserving']
                2           ['Constant', 'Constant']
                3           ['None', 'Constant']
                4           ['Constant', 'None']
                5           ['Additive', 'Additive']
                6           ['Constant', 'Additive']
                7           ['Additive', 'Constant']
                8           ['Multiplicative', 'Multiplicative']
                9           ['Constant', 'Multiplicative']
                10          ['Multiplicative', 'Constant']
                =========== ====================================

                =========== ====================================
                    2D Symbolic Patterns Possible Combinations
                ------------------------------------------------
                index       pattern combination
                =========== ====================================
                0           ['Order_Preserving', 'None']
                1           ['None', 'Order_Preserving']
                2           ['Constant', 'Constant']
                3           ['None', 'Constant']
                4           ['Constant', 'None']
                =========== ====================================

                =========== ======================================================
                        3D Numeric Patterns Possible Combinations
                ------------------------------------------------------------------
                index       pattern combination
                =========== ======================================================
                0           ['Order_Preserving', 'None', 'None']
                1           ['None', 'Order_Preserving', 'None']
                2           ['None', 'None', 'Order_Preserving']
                3           ['Constant', 'Constant', 'Constant']
                4           ['None', 'Constant', 'Constant']
                5           ['Constant', 'Constant', 'None']
                6           ['Constant', 'None', 'Constant']
                7           ['Constant', 'None', 'None']
                8           ['None', 'Constant', 'None']
                9           ['None', 'None', 'Constant']
                10          ['Additive', 'Additive', 'Additive']
                11          ['Additive', 'Additive', 'Constant']
                12          ['Constant', 'Additive', 'Additive']
                13          ['Additive', 'Constant', 'Additive']
                14          ['Additive', 'Constant', 'Constant']
                15          ['Constant', 'Additive', 'Constant']
                16          ['Constant', 'Constant', 'Additive']
                17          ['Multiplicative', 'Multiplicative', 'Multiplicative']
                18          ['Multiplicative', 'Multiplicative', 'Constant']
                19          ['Constant', 'Multiplicative', 'Multiplicative']
                20          ['Multiplicative', 'Constant', 'Multiplicative']
                21          ['Multiplicative', 'Constant', 'Constant']
                22          ['Constant', 'Multiplicative', 'Constant']
                23          ['Constant', 'Constant', 'Multiplicative']
                =========== ======================================================

                =========== ======================================================
                        3D Numeric Patterns Possible Combinations
                ------------------------------------------------------------------
                index       pattern combination
                =========== ======================================================
                0           ['Order_Preserving', 'None', 'None']
                1           ['None', 'Order_Preserving', 'None']
                2           ['None', 'None', 'Order_Preserving']
                3           ['Constant', 'Constant', 'Constant']
                4           ['None', 'Constant', 'Constant']
                5           ['Constant', 'Constant', 'None']
                6           ['Constant', 'None', 'Constant']
                7           ['Constant', 'None', 'None']
                8           ['None', 'Constant', 'None']
                9           ['None', 'None', 'Constant']
                =========== ======================================================

        bktype: {'NORMAL', 'UNIFORM', 'DISCRETE', 'MISSING'}, default 'UNIFORM'
            Determines the distribution used to generate the background values.
        clusterdistribution: list or array, default [['UNIFORM', 4.0, 4.0], ['UNIFORM', 4.0, 4.0]]
            Distribution used to calculate the size of a cluster.

            **Shape**: number of dimensions, 3 -> param1(str), param2(float), param3(float)

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

            **Range**: [0,1]
        maxclustsperoverlappedarea: int, default 0
            Maximum number of clusters overlapped per area. Maximum number of clusters that can overlap together.

            Not used if plaidcoherency == 'NO_OVERLAPPING'.

            **Range**: [0, nclusters]
        maxpercofoverlappingelements: float, default 0.0
            Maximum percentage of values shared by overlapped clusters.

            Not used if plaidcoherency == 'NO_OVERLAPPING'.

            **Range**: [0,1]
        percofoverlappingrows: float, default 1.0
            Percentage of allowed amount of overlaping across clusters rows.

            Not used if plaidcoherency == 'NO_OVERLAPPING'.

            **Range**: [0,1]
        percofoverlappingcolumns: float, default 1.0
            Percentage of allowed amount of overlaping across clusters columns.

            Not used if plaidcoherency == 'NO_OVERLAPPING'.

            **Range**: [0,1]
        percofoverlappingcontexts: float, default 1.0
            Percentage of allowed amount of overlaping across clusters contexts.

            Not used if plaidcoherency == 'NO_OVERLAPPING' or cuda >= 3.

            **Range**: [0,1]
        percmissingsonbackground: float, 0.0
            Percentage of missing values on the background, that is, values that do not belong to planted clusters.

            **Range**: [0,1]
        percmissingsonclusters: float, 0.0
            Maximum percentage of missing values on each cluster.

            **Range**: [0,1]
        percnoiseonbackground: float, 0.0
            Percentage of noisy values on background, that is, values with added noise.

            **Range**: [0,1]
        percnoiseonclusters: float, 0.0
            Maximum percentage of noisy values on each cluster.

            **Range**: [0,1]
        percnoisedeviation: int or float, 0.0
            Percentage of symbol on noisy values deviation, that is, the maximum difference between the current symbol
            on the matrix and the one that will replaced it to be considered noise.

            If dstype == Numeric then percnoisedeviation -> float else int.

            **Ex**: Let Alphabet = [1,2,3,4,5] and CurrentSymbol = 3, if the noiseDeviation is '1', then CurrentSymbol
            will be, randomly, replaced by either '2' or '4'. If noiseDeviation is '2', CurrentSymbol can be replaced by
            either '1','2','4' or '5'.
        percerroesonbackground: float, 0.0
            Percentage of error values on background. Similar as noise, a new value is considered an error if the
            difference between it and the current value in the matrix is greater than noiseDeviation.

            **Ex**: Alphabet = [1,2,3,4,5], If currentValue = 2, and errorDeviation = 2, to turn currentValue an error,
            it's value must be replaced by '5', that is the only possible value that respects
            abs(currentValue - newValue) > noiseDeviation

            **Range**: [0,1]
        percerrorsonclusters: float, 0.0
            Percentage of errors values on background. Similar as noise, a new value is considered an error if the
            difference between it and the current value in the matrix is greater than noiseDeviation.

            **Ex**: Alphabet = [1,2,3,4,5], If currentValue = 2, and errorDeviation = 2, to turn currentValue an error,
            it's value must be replaced by '5', that is the only possible value that respects
            abs(currentValue - newValue) > noiseDeviation

            **Range**: [0,1]
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

            **Shape**: alphabets length
        nsymbols: int, default 10
            Defines the length of the alphabet, instead of defining specific symbols this parameter can be passed, and
            a list of strings will be create with range(1, cuda), where cuda represents this parameter.

            Only used if dstype == 'SYMBOLIC' and symbols is None.
        mean: int or float, default 14.0
            Mean for the background's distribution. Only used when bktype == 'NORMAL'.
        stdev: int or float, default 7.0
            Standard deviation for the background's distribution. Only used when bktype == 'NORMAL'.
        probs: list or array of floats
            Background weighted distribution probabilities. Only used when bktype == 'DISCRETE'.
            No default probabilities, if probs is None and bktype == 'DISCRETE', bktype defaults to 'UNIFORM'.

            **Shape**: Number of symbols or possible integers

            **Range**: [0,1]

            **Math**: sum(probs) == 1
        in_memory: bool, default None
            Determines if generated datasets return dense or sparse matrix (True/False).

            If None then if the generated dataset's size is larger then 10**5 it defaults to sparse, else outputs dense.

            Note
            ----
                This parameter can be overwritten in the generate method.

        Attributes
        ----------

        _n: int
            Dimensionality.
        _stdout: System object (java)
            default System.out
        dstype: {'NUMERIC', 'SYMBOLIC'}
            Type of Dataset to be generated, numeric or symbolic(categorical).
        patterns: list
            Type of patterns that will be hidden in the data.
        clusterdistribution: list
            Distribution used to calculate the size of a cluster.
        contiguity: {'COLUMNS', 'CONTEXTS', 'NONE'}
            Data contiguity.
        time_profile: {'RANDOM', 'MONONICALLY_INCREASING', 'MONONICALLY_DECREASING', None}
            Time profile for the ORDER_PRESERVING pattern.
        seed: int
            Seed to initialize random objects.
        realval: bool
            If the dataset is real valued.
        minval: float
            Dataset's minimum value.
        maxval: float
             Dataset's maximum value.
        noise: tuple
            Dataset's noise settings.
        errors: tuple
            Dataset's error settings.
        missing: tuple
            Dataset's missing settings.
        symbols: list
            Dataset's alphabet.
        nsymbols: int
            Length of the alphabet.
        plaidcoherency: {'ADDITIVE', 'MULTIPLICATIVE', 'INTERPOLED', 'NONE', 'NO_OVERLAPPING'}
            Type of plaid coherency.
        percofoverlappingclusts: float
            Percentage of overlapping clusters.
        maxclustsperoverlappedarea: int
            Maximum number of clusters overlapped per area.
        maxpercofoverlappingelements: float
            Maximum percentage of values shared by overlapped clusters.
        percofoverlappingrows: float
            Percentage of allowed amount of overlaping across clusters rows.
        percofoverlappingcolumns: float
            Percentage of allowed amount of overlaping across clusters columns.
        percofoverlappingcontexts: float
            Percentage of allowed amount of overlaping across clusters contexts.
        background: list
            Dataset's background settings
        generatedDataset: Dataset object (java)
            Generated dataset.
        X: dense or sparse tensor
            Generated dataset as tensor.
        Y: list
            Hidden cluster labels.
        graph: Graph object
            N-partite graph
        in_memory: bool
            If dataset should be saved in memory (dense format)
        silenced: bool
            If prints to the console.

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

    def _start_silencing(self, silence=None):

        """
        Starts silencing all java prints to terminal.

        Parameters
        ----------

        silence: bool, default None
            If True all java prints to terminal are ignored.
            If None, defaults to class attribute's value.
        """

        if silence is None:
            silence = self.silenced

        if bool(silence):
            System.setOut(PrintStream('logs'))

    def _stop_silencing(self):

        """
        Stops silencing all java prints to terminal.
        """

        System.setOut(self._stdout)

        try:
            os.remove('logs')
        except FileNotFoundError:
            pass

    def get_params(self):

        """
        Returns the classes attributes.

        Returns
        -------

        dict
            Values of class attributes.

        Examples
        --------

        >>> generator = BiclusterGenerator()
        >>> generator.get_params()
        {'X': None, 'Y': None, 'background': ['UNIFORM'], 'clusterdistribution': [['UNIFORM', 4.0, 4.0],
        ['UNIFORM', 4.0, 4.0]], 'contiguity': 'NONE', 'dstype': 'NUMERIC', 'errors': (0.0, 0.0, 0.0),
        'generatedDataset': None, 'graph': None, 'in_memory': None, 'maxclustsperoverlappedarea': 0,
        'maxpercofoverlappingelements': 0.0, 'maxval': 10.0, 'minval': -10.0, 'missing': (0.0, 0.0),
        'noise': (0.0, 0.0, 0.0), 'patterns': [['CONSTANT', 'CONSTANT']], 'percofoverlappingclusts': 0.0,
        'percofoverlappingcolumns': 1.0, 'percofoverlappingcontexts': 1.0, 'percofoverlappingrows': 1.0,
        'plaidcoherency': 'NO_OVERLAPPING', 'realval': True, 'seed': -1, 'silenced': False, 'time_profile': None}

        """

        attributes = inspect.getmembers(self, lambda a: not (inspect.isroutine(a)))

        return {a[0]: a[1]
                for a in attributes if not (a[0].startswith('__') and a[0].endswith('__') or a[0].startswith('_'))
                }

    @property
    def cluster_info(self):

        """
        Returns clusters info.

        Returns
        -------

        dict
            Hidden cluster info.

        Examples
        --------
        >>> generator = BiclusterGenerator(silence=True)
        >>> generator.generate(no_return=True)
        >>> generator.cluster_info
        {'0': {'%Errors': '0', 'Type': 'Numeric', '%Missings': '0', '%Noise': '0', 'X': [15, 51, 63, 92],
        'Y': [7, 29, 35, 94], 'RowPattern': 'Constant', 'ColumnPattern': 'Constant',
        'Data': [['-8.61', '-8.61', '-8.61', '-8.61'], ['-8.61', '-8.61', '-8.61', '-8.61'],
        ['-8.61', '-8.61', '-8.61', '-8.61'], ['-8.61', '-8.61', '-8.61', '-8.61']],
        'PlaidCoherency': 'No Overlapping', '#rows': 4, '#columns': 4}}

        """

        if self.generatedDataset is None:
            return dict()

        cluster_type = {2: 'bi', 3: 'Tri'}[self._n]
        geninfo_params = {2: [self.generatedDataset, False], 3: [self.generatedDataset]}[self._n]

        js = json.loads(
            str(getattr(self.generatedDataset, 'get{}csInfoJSON'.format(cluster_type.capitalize()))
                (*geninfo_params).getJSONObject('{}clusters'.format(cluster_type)).toString())
        )

        return js

    @property
    def coverage(self):

        """
        Returns clusters dataset coverage.

        Returns
        -------

        float
            Percentage of cluster coverage.

        Examples
        --------
        >>> generator = BiclusterGenerator(silence=True)
        >>> generator.generate(no_return=True)
        >>> generator.coverage
        0.16

        """
        if self.generatedDataset is None:
            return float(0)

        return ((self.generatedDataset.getSize() - self.generatedDataset.getBackgroundSize()) /
                self.generatedDataset.getSize()) * 100

    @abc.abstractmethod
    def _initialize_seed(self):

        """
        Uses seed attribute to initialize random object.
        """
        pass

    @abc.abstractmethod
    def _build_background(self):

        """
        Builds a Background object, using the background attribute.

        Returns
        -------

        Background object
            Dataset's background.
        """
        pass

    @abc.abstractmethod
    def _build_generator(self, class_call, params, contexts_index):
        """
        Builds the (Java)Generator object.

        Parameters
        ----------

        class_call: {'NumericDatasetGenerator', 'SymbolicDatasetGenerator'}
            Name of generator to initialize.
        params: list
            Parameters to initialize generator.
        contexts_index: int
            Position of the ncontext param. Only used when dimensionality < 2.

        Returns
        -------

        (Java)Generator object
            Generator for data generation.
        """
        pass

    @abc.abstractmethod
    def _build_patterns(self):

        """
        Builds a list of pattern objects, using the patterns, time_profile, and dstype attributes.

        Returns
        -------

        ArrayList
            List of pattern objects.
        """
        pass

    @abc.abstractmethod
    def _build_structure(self):

        """
        Builds a Structure object, using the clusterdistribution and contiguity attributes.

        Returns
        -------

        Structure object
            Dataset's structure.
        """
        pass

    @abc.abstractmethod
    def _build_overlapping(self):

        """
        Builds an OverlappingSettings object.

        Returns
        -------

        OverlappingSettings object
            Dataset's overlapping settings.
        """
        pass

    @abc.abstractmethod
    def save(self, extension='default', file_name='example_dataset', path=None, single_file=None, **kwargs):

        """
        Saves data files to chosen path.

        Parameters
        ----------

        extension: {'default', 'csv'}, default 'default'
            Extension of saved data file.
        file_name: str, default 'example_dataset'
            Saved files prefix.
        path: str, default None
            Path to save files. If None then files are saved in the current working directory.
        single_file: Bool, default None.
            If False dataset is saved in multiple data files. If None then if the dataset's size is larger then 10**5
            it defaults to False, else True. Only used if extension=='default'.
        **kwargs: any, default None
            Additional keywords that are passed on.

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _java_to_numpy(generatedDataset):

        """
        Extracts numpy array from Dataset object.

        Parameters
        ----------

        generatedDataset: Dataset object
            Generated dataset (java object).

        Returns
        -------

        numpy array
            Generated dataset as numpy array (dense tensor).

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _java_to_sparse(generatedDataset):

        """
        Extracts sparce tensor from Dataset object.

        Parameters
        ----------

        generatedDataset: Dataset object
            Generated dataset (java object).

        Returns
        -------

        Sparse tensor
            Generated dataset as sparse tensor.

        """
        pass

    def to_tensor(self, generatedDataset=None, in_memory=None, keys=None):

        """
        Returns generated dataset as somekind of tensor and hidden cluster labels.

        Parameters
        ----------

        generatedDataset: Dataset object
            Generated dataset (java object).
        in_memory: bool, default None
            Determines if generated datasets return dense or sparse matrix (True/False).

            If None then if the generated dataset's size is larger then 10**5 it defaults to sparse, else outputs dense.
        keys: list, default ['X', 'Y', 'Z']
            Axis keys. Do not overwrite, unless you are using a different dataset object.

        Returns
        -------

        dense or sparse tensor
            Generated dataset as tensor.

            **Shape**: (ncontexts, nrows, ncols) or (nrows, ncols)
        list
            Hidden cluster labels.

            **Shape**: (nclusters, any)

        Examples
        --------
        >>> generator = BiclusterGenerator(silence=True)
        >>> generator.generate(no_return=True)
        >>> x, y = generator.to_tensor(generatedDataset=generator.generatedDataset, in_memory=True)
        >>> x
        array([[-4.15,  9.88,  7.69, ...,  3.68,  1.72, -6.95],
               [ 7.37,  2.63, -0.13, ..., -2.53,  2.03,  8.03],
               [ 4.28,  0.36,  8.66, ..., -1.11,  6.28, -1.03],
               ...,
               [-9.25, -9.15, -4.68, ...,  2.06, -6.19,  2.54],
               [ 2.63, -3.03,  3.8 , ...,  4.13, -4.17,  7.68],
               [-1.98,  8.02,  1.89, ...,  3.59,  4.27,  6.4 ]])

        """

        self._start_silencing()

        if generatedDataset is None:
            generatedDataset = self.generatedDataset

        if in_memory is None:
            in_memory = self._asses_memory(gends=generatedDataset)

        if keys is None:
            keys = ['X', 'Y', 'Z']

        # Get Tensor

        if bool(in_memory):
            self.X = self._java_to_numpy(generatedDataset)

        else:
            self.X = self._java_to_sparse(generatedDataset)

        # Get clusters

        js = self.cluster_info

        keys = keys[:self._n]

        self.Y = [[js[i][key] for key in keys] for i in js.keys()]

        self._stop_silencing()

        return self.X, self.Y

    @staticmethod
    @abc.abstractmethod
    def _dense_to_dgl(x, device, cuda=0, nclusters=1, clust_init='zeros'):

        """
        Extracts a partite dgl graph from a numpy array

        Parameters
        ----------

        x: numpy array
            Data array.
        device: {'cpu', 'gpu'}
            Type of device for storing the tensor.
        cuda: int, default 0
            Index of cuda device to use. Only used if device==True.

        Returns
        -------

        heterograph object
            numpy array as n-partite dgl graph, where n==dim.

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def _dense_to_networkx(x, **kwargs):

        """
        Extracts a partite networkx graph from numpy array

        Parameters
        ----------

        x: numpy array
            Data array.
        **kwargs: any, default None
            Additional keywords have no effect but might be accepted for compatibility.

        Returns
        -------

        Graph object
            numpy array as n-partite networkx graph, where n==dim.

        """
        pass

    def to_graph(self, x=None, framework='networkx', device='cpu', **kwargs):

        """
        Returns a n-partite graph, where n==dim.

        Parameters
        ----------

        x: numpy array
            Data array.
        framework: {networkx, dgl}, default 'networkx'
            Backend to use to build graph.
        device: {'cpu', 'gpu'}, default 'cpu'
            Type of device for storing the tensor. Only used if framework==dgl.
        **kwargs: any, default None
            Additional keywords that are passed on.

        Returns
        -------

        Graph object
            N-partite graph, where n==dim.

            **Shape**: (nrows + ncols + ncontexts, nrows * ncols * ncontexts * 3(dim)) or
            (nrows + ncols, nrows * ncols)

        Examples
        --------

        >>> generator = BiclusterGenerator(silence=True)
        >>> X, y = generator.generate()
        >>> g = generator.to_graph(X, framework='dgl')
        Graph(num_nodes={'col': 100, 'row': 100},
              num_edges={('row', 'elem', 'col'): 10000},
              metagraph=[('row', 'col', 'elem')])

        """

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
            if isinstance(x, COO) or isinstance(x, csr_matrix):
                x = x.todense()

                if isinstance(x, np.matrix):
                    x = np.array(x)

            if device == 'gpu' and not th.cuda.is_available():

                device = 'cpu'

                warnings.warn('CUDA not available CPU will be used instead')

            if device == 'gpu' and framework == 'networkx':

                framework = 'dgl'

                warnings.warn('The Networkx library is not compatible with gpu devices. '
                              'DGL will be used instead.')

            # call private method
            self.graph = getattr(self, '_dense_to_{}'.format(framework))(x, device=device, **kwargs)

            return self.graph

        else:
            raise AttributeError('No generated dataset exists. '
                                 'Data must first be generated using the .generate() method.')

    def _get_dstype_vars(self, nrows, ncols, ncontexts, nclusters, background):

        """
        Prepares parameters to initialize the generator.

        Parameters
        ----------

        nrows: int
            Number of rows in generated dataset.
        ncols: int
            Number of columns in generated dataset.
        ncontexts: int
            Number of contexts in generated dataset.
        nclusters: int
            Number of clusters in generated dataset.
        background: Background object
            Dataset's background.

        Returns
        -------

        str
            Name of generator to initialize.
        list
            Parameters to initialize generator.
        int
            Position of the ncontext param.
        """

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

    def _asses_memory(self, in_memory=None, **kwargs):

        """
        Returns True if dataset should be saved in memory.

        Parameters
        ----------

        in_memory: bool, default None
            Determines if dataset should be saved in memory.

            If None then if the gends > 10**5 it defaults to False.
        gends: Dataset object, default None
            Generated Dataset (java object).

            Only used if in_memory is None, in that case gends cannot be None.

        Returns
        -------

        bool
            True if dataset should be saved in memory, else False.
        """

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

    def _plant_quality_settings(self, generatedDataset):

        """
        Plants quality settings on generated dataset

        Parameters
        ----------

        generatedDataset: Dataset object
            Generated dataset (java object).
        """

        generatedDataset.plantMissingElements(*self.missing)
        generatedDataset.plantNoisyElements(*self.noise)
        generatedDataset.plantErrors(*self.errors)

    def generate(self, nrows=100, ncols=100, ncontexts=3, nclusters=1, no_return=False, **kwargs):

        """
        Generates dataset, and may return somekind of tensor and hidden cluster labels.

        Parameters
        ----------

        nrows: int, default 100
            Number of rows in generated dataset.
        ncols: int, default 100
            Number of columns in generated dataset.
        ncontexts: int, default 3
            Number of contexts in generated dataset.
            Only used if dim >= 3.
        nclusters: int, default 1
            Number of clusters in generated dataset.
        no_return: bool, default False
            If True method returns None.
        **kwargs: any, default None
            Additional keywords that are passed on.

        Returns
        -------

        dense or sparse tensor
            Generated dataset as tensor.

            **Shape**: (ncontexts, nrows, ncols) or (nrows, ncols)
        list
            Hidden cluster labels.

            **Shape**: (nclusters, any)
        None
            If no_return==True.

        Examples
        --------

        >>> gen = BiclusterGenerator(silence=True)
        >>> x, y = gen.generate(nrows=100, ncols=200, nclusters=20, in_memory=True)
        >>> x
        array([[-7.36,  4.88,  8.42, ..., -5.04, -4.93,  6.35],
               [-7.1 ,  0.47, -2.58, ..., -3.03,  0.42,  8.76],
               [-8.08,  4.19,  2.53, ..., -4.3 ,  7.54,  0.94],
               ...,
               [-0.52,  0.38,  6.98, ..., -7.6 ,  5.71,  9.24],
               [-1.28, -3.55, -3.13, ..., -4.17, -6.05, -9.87],
               [-5.79, -6.05, -2.24, ...,  1.88,  1.97,  6.05]])

        """

        # Enforce Types
        nrows = int(nrows)
        ncols = int(ncols)
        ncontexts = int(ncontexts)
        nclusters = int(nclusters)
        no_return = bool(no_return)
        
        self._start_silencing()

        # initialize random seed
        self._initialize_seed()

        # define background
        background = self._build_background()

        # initialise data generator
        params = self._get_dstype_vars(nrows, ncols, ncontexts, nclusters, background)

        generator = self._build_generator(*params)

        # get patterns
        patterns = self._build_patterns()

        # get structure
        structure = self._build_structure()

        # get overlapping
        overlapping = self._build_overlapping()

        # generate dataset
        generatedDataset = generator.generate(patterns, structure, overlapping)

        # plant missing values, noise & errors
        self._plant_quality_settings(generatedDataset)

        # return
        self.generatedDataset = generatedDataset

        self._stop_silencing()

        if no_return:
            return None, None

        return self.to_tensor(generatedDataset, **kwargs)

    @staticmethod
    def shutdownJVM():

        """
        Shuts down JVM.

        Caution
        -------
            If the JVM is shutdown it cannot be restarted on the same session.
        """

        try:
            jpype.shutdownJVM()
        except RuntimeError:
            pass


