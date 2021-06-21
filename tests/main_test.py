import networkx

from nclustgen import \
    BiclusterGenerator as bg, \
    BiclusterGeneratorbyConfig as bgconfig, \
    TriclusterGenerator as tg, \
    TriclusterGeneratorbyConfig as tgconfig

import unittest
import pathlib as pl
import numpy
from scipy.sparse import csr_matrix
from sparse import COO
import os
import json
from networkx import Graph
import torch as th
import dgl

from java.lang import System
from java.io import PrintStream

from com.gbic.types import Background as bic_background
from com.gtric.types import Background as tric_background

from com.gbic.utils import SingleBiclusterPattern
from com.gtric.utils import TriclusterPattern

from com.gbic.utils import BiclusterStructure
from com.gtric.utils import TriclusterStructure

from com.gbic.utils import OverlappingSettings as overlap_bic
from com.gtric.utils import OverlappingSettings as overlap_tric

from com.gbic.domain.dataset import NumericDataset as bic_n_dataset
from com.gbic.domain.dataset import SymbolicDataset as bic_s_dataset

from com.gtric.domain.dataset import NumericDataset as tric_n_dataset
from com.gtric.domain.dataset import SymbolicDataset as tric_s_dataset

from com.gbic import generator as gen_bic
from com.gtric import generator as gen_tric


class TestCaseBase(unittest.TestCase):

    @staticmethod
    def assertIsFile(path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

    @staticmethod
    def assertIsNotFile(path):
        if pl.Path(path).resolve().is_file():
            raise AssertionError("File exists: %s" % str(path))


# TODO test getters
class GenTest(TestCaseBase):

    def test_silence(self):

        # start silenced and un-silenced objects
        instance_T = bg(silence=True)
        instance_F = bg(silence=False)

        # check objects params
        self.assertTrue(instance_T.silenced)
        self.assertFalse(instance_F.silenced)

        self.assertIsNotNone(instance_T._stdout)
        self.assertIsNotNone(instance_F._stdout)

        # enforce System default status
        System.setOut(instance_T._stdout)

        # check method logic

        instance_T._start_silencing()
        self.assertIsFile('logs')

        instance_T._stop_silencing()
        self.assertIsNotFile('logs')
        self.assertEqual(System.out, instance_T._stdout)

        instance_T._start_silencing(silence=False)
        self.assertIsNotFile('logs')
        self.assertEqual(System.out, instance_T._stdout)

        instance_F._start_silencing()
        self.assertIsNotFile('logs')
        self.assertEqual(System.out, instance_T._stdout)

    def test_memory(self):

        # start in_memory, off_memory, and undefined objects
        instance_T = bg(in_memory=True)
        instance_F = bg(in_memory=False)
        instance_N = bg()

        # check objects params
        self.assertTrue(instance_T.in_memory)
        self.assertFalse(instance_F.in_memory)
        self.assertIsNone(instance_N.in_memory)

        # create proxy gends class

        class gends:

            def __init__(self, shape):

                self.shape = shape

            def getNumRows(self):
                return self.shape[0]

            def getNumCols(self):
                return self.shape[1]

        gends_T = gends((100, 100))
        gends_F = gends((1000, 1000))

        # check method logic
        self.assertTrue(instance_T._asses_memory())
        self.assertTrue(instance_T._asses_memory(in_memory=True))
        self.assertFalse(instance_T._asses_memory(in_memory=False))

        self.assertTrue(instance_T._asses_memory(gends=gends_T))
        self.assertTrue(instance_T._asses_memory(gends=gends_F))

        self.assertFalse(instance_F._asses_memory())
        self.assertTrue(instance_F._asses_memory(in_memory=True))
        self.assertFalse(instance_F._asses_memory(in_memory=False))

        self.assertFalse(instance_F._asses_memory(gends=gends_T))
        self.assertFalse(instance_F._asses_memory(gends=gends_F))

        with self.assertRaises(AttributeError):
            instance_N._asses_memory()

        self.assertTrue(instance_N._asses_memory(in_memory=True))
        self.assertFalse(instance_N._asses_memory(in_memory=False))

        self.assertTrue(instance_N._asses_memory(gends=gends_T))
        self.assertFalse(instance_N._asses_memory(gends=gends_F))

    def test_properties(self):

        instance = bg(silence=True)
        instance.generate()

        self.assertTrue(isinstance(instance.get_params(), dict))
        self.assertTrue(isinstance(instance.cluster_info, dict))
        self.assertTrue(isinstance(instance.coverage, float))


class BicsGenTest(TestCaseBase):

    def setUp(self):

        self.integration = True

    def test_background(self):

        # Test initialization

        probs = [0.5, 0.25, 0.25]

        instance_numeric_uniform = bg(silence=True)
        instance_numeric_missing = bg(bktype='Missing', silence=True)
        instance_numeric_discrete = bg(bktype='Discrete', minval=1, maxval=3, probs=probs, silence=True)
        instance_numeric_normal = bg(bktype='Normal', silence=True)

        instance_symbolic_uniform = bg(dstype='SYMBOLIC', silence=True)
        instance_symbolic_missing = bg(dstype='symbolic', bktype='Missing', silence=True)
        instance_symbolic_discrete = bg(dstype='Symbolic', bktype='Discrete', nsymbols=3, probs=probs, silence=True)
        instance_symbolic_discrete_noprobs = bg(dstype='Symbolic', bktype='Discrete', nsymbols=3, silence=True)
        instance_symbolic_normal = bg(dstype='Symbolic', bktype='Normal', silence=True)

        # test initialization

        self.assertEqual(instance_numeric_uniform.background, ['UNIFORM'])
        self.assertEqual(instance_numeric_missing.background, ['MISSING'])
        self.assertEqual(instance_numeric_discrete.background, ['DISCRETE', probs])
        self.assertEqual(instance_numeric_normal.background, ['NORMAL', 14, 7])

        self.assertEqual(instance_symbolic_uniform.background, ['UNIFORM'])
        self.assertEqual(instance_symbolic_missing.background, ['MISSING'])
        self.assertEqual(instance_symbolic_discrete.background, ['DISCRETE', probs])
        self.assertEqual(instance_symbolic_discrete_noprobs.background, ['UNIFORM'])
        self.assertEqual(instance_symbolic_normal.background, ['NORMAL', 14, 7])

        # test method

        self.assertTrue(isinstance(instance_numeric_uniform._build_background(), bic_background))
        self.assertTrue(isinstance(instance_numeric_missing._build_background(), bic_background))
        self.assertTrue(isinstance(instance_numeric_discrete._build_background(), bic_background))
        self.assertTrue(isinstance(instance_numeric_normal._build_background(), bic_background))

        self.assertTrue(isinstance(instance_symbolic_uniform._build_background(), bic_background))
        self.assertTrue(isinstance(instance_symbolic_missing._build_background(), bic_background))
        self.assertTrue(isinstance(instance_symbolic_discrete._build_background(), bic_background))
        self.assertTrue(isinstance(instance_symbolic_discrete_noprobs._build_background(), bic_background))
        self.assertTrue(isinstance(instance_symbolic_normal._build_background(), bic_background))

        # integration

        if self.integration:

            instance_numeric_uniform.generate()
            instance_numeric_missing.generate()
            instance_numeric_discrete.generate()
            instance_numeric_normal.generate()

            instance_symbolic_uniform.generate()
            instance_symbolic_missing.generate()
            instance_symbolic_discrete.generate()
            instance_symbolic_discrete_noprobs.generate()
            instance_symbolic_normal.generate()

    def test_patterns(self):

        patterns = [
            ['Numeric', [['Constant', 'CONSTANT'], ['CONSTANT', 'none']], None],
            ['Symbolic', [['NONE', 'ORDER_PRESERVING']], 'Random'],
            ['Numeric', [['NONE', 'ADDITIVE']], None],
            ['NUMERIC', [['multiplicative', 'CONSTANT']], None]
        ]

        expected_patterns = [
            ['NUMERIC', [['CONSTANT', 'CONSTANT'], ['CONSTANT', 'NONE']], None],
            ['SYMBOLIC', [['NONE', 'ORDERPRESERVING']], 'RANDOM'],
            ['NUMERIC', [['NONE', 'ADDITIVE']], None],
            ['NUMERIC', [['MULTIPLICATIVE', 'CONSTANT']], None]
        ]

        for i, (pattern, expected) in enumerate(zip(patterns, expected_patterns)):

            # print(i)

            # build instance and tester
            instance = bg(dstype=pattern[0], patterns=pattern[1], timeprofile=pattern[2], silence=True)
            builts = instance._build_patterns()

            for built, expect in zip(builts, expected[1]):

                # check class
                self.assertTrue(isinstance(built, SingleBiclusterPattern))

                # check patterns
                self.assertEqual(str(built.getBiclusterType().toString()).upper(), expected[0])
                self.assertEqual(str(built.getRowsPattern().toString()).upper(), expect[0])
                self.assertEqual(str(built.getColumnsPattern().toString()).upper(), expect[1])

                # check time profile
                if expected[2]:
                    self.assertEqual(str(built.getTimeProfile().toString()).upper(), expected[2])
                else:
                    self.assertIsNone(built.getTimeProfile())

            # check integration

            if self.integration:
                instance.generate()

    def test_struture(self):

        distribution = [
            [[['UNIFoRM', 1.0, 4], ['UNIFORM', 1, 4]], 'NONE'],
            [[['Normal', 3, 1], ['UNIFORM', 1, 4]], 'CoLUMNS'],
            [[['Normal', 3, 1], ['UNIFORM', 1, 4]], 'CONTEXTS']
        ]

        expected_distribution = [
            [[['UNIFORM', 1, 4], ['UNIFORM', 1, 4]], 'NONE'],
            [[['NORMAL', 3, 1], ['UNIFORM', 1, 4]], 'COLUMNS'],
            [[['NORMAL', 3, 1], ['UNIFORM', 1, 4]], 'NONE']
        ]

        for i, (dist, expected) in enumerate(zip(distribution, expected_distribution)):

            # build instance and tester
            instance = bg(clusterdistribution=dist[0], contiguity=dist[1], silence=True)
            built = instance._build_structure()

            # check class
            self.assertTrue(isinstance(built, BiclusterStructure))

            # check row distribution
            self.assertEqual(str(built.getRowsDistribution().toString()), expected[0][0][0])
            self.assertEqual(int(built.getRowsParam1()), expected[0][0][1])
            self.assertEqual(int(built.getRowsParam2()), expected[0][0][2])

            # check cols distribution
            self.assertEqual(str(built.getColumnsDistribution().toString()), expected[0][1][0])
            self.assertEqual(int(built.getColumnsParam1()), expected[0][1][1])
            self.assertEqual(int(built.getColumnsParam2()), expected[0][1][2])

            # check row contiguity
            self.assertEqual(str(built.getContiguity().toString()), expected[1])

            # check integration
            if self.integration:
                instance.generate()

    def test_generator(self):

        generator_parameters = [
            {
                'dstype': 'Numeric',
                'realval': False,
                'minval': 1,
                'maxval': 5,
                'silence': True
            },
            {
                'dstype': 'Numeric',
                'realval': True,
                'minval': 1,
                'maxval': 5,
                'silence': True
            },
            {
                'dstype': 'Symbolic',
                'nsymbols': 5,
                'symmetries': 1,
                'silence': True
            }
        ]

        expected_parameters = [
            {
                'dstype': 'NUMERIC',
                'realval': False,
                'minval': 1,
                'maxval': 5,
                'silence': True
            },
            {
                'dstype': 'NUMERIC',
                'realval': True,
                'minval': 1,
                'maxval': 5,
                'silence': True
            },
            {
                'dstype': 'SYMBOLIC',
                'nsymbols': 5,
                'symmetries': 1,
                'silence': True
            }
        ]

        for i, (params, expected) in enumerate(zip(generator_parameters, expected_parameters)):

            # build instance
            instance = bg(**params)

            background = instance._build_background()
            generate_params = [100, 100, 1, 5, background]

            parameters = instance._get_dstype_vars(*generate_params)
            built = instance._build_generator(*parameters)

            generate_params.remove(1)

            self.assertTrue(isinstance(built, getattr(gen_bic, parameters[0])))

            if expected['dstype'] == 'NUMERIC':

                self.assertEqual(parameters[1][1:5], generate_params)
                self.assertEqual(parameters[1][0], expected['realval'])
                self.assertEqual(parameters[1][5], expected['minval'])
                self.assertEqual(parameters[1][6], expected['maxval'])
                self.assertEqual(parameters[2], 3)

            else:
                self.assertEqual(parameters[1][:4], generate_params)
                self.assertEqual(len(parameters[1][4]), expected['nsymbols'])
                self.assertEqual(parameters[1][5], expected['symmetries'])
                self.assertEqual(parameters[2], 2)

            # check integration
            if self.integration:
                instance.generate()

    def test_overlapping(self):

        overlapping = [
            ['NO_OVeRLaPPING', 0, 0.0, 0.0, 1.0, 1.0],
            [None, 0, 0, 0.0, 1.0, 1.0],
            ['AddITIVE', 0.5, 2, 0.5, 0.75, 0.5],
            ['MULTiPLICATIVE', 0.25, 2.0, 0.75, 1.0, 1.0],
            ['INTERPOLED', 0.75, 3, 1, 1, 1],
        ]

        expected_overlapping = [
            ['NO OVERLAPPING', 0.0, 0, 0.0, 1.0, 1.0],
            ['NONE', 0.0, 0, 0.0, 1.0, 1.0],
            ['ADDITIVE', 0.5, 2, 0.5, 0.75, 0.5],
            ['MULTIPLICATIVE', 0.25, 2, 0.75, 1.0, 1.0],
            ['INTERPOLED', 0.75, 3, 1.0, 1.0, 1.0],
        ]

        for i, (overlap, expected) in enumerate(zip(overlapping, expected_overlapping)):
            # build instance and tester
            instance = bg(
                plaidcoherency=overlap[0],
                percofoverlappingclusters=overlap[1],
                maxclustsperoverlappedarea=overlap[2],
                maxpercofoverlappingelements=overlap[3],
                percofoverlappingrows=overlap[4],
                percofoverlappingcolumns=overlap[5],
                silence=True
            )

            built = instance._build_overlapping()

            # check class
            self.assertTrue(isinstance(built, overlap_bic))

            # check values

            self.assertEqual(str(built.getPlaidCoherency().toString()).upper(), expected[0])
            self.assertEqual(float(built.getPercOfOverlappingBics()), expected[1])
            self.assertEqual(int(built.getMaxBicsPerOverlappedArea()), expected[2])
            self.assertEqual(float(built.getMaxPercOfOverlappingElements()), expected[3])
            self.assertEqual(float(built.getPercOfOverlappingRows()), expected[4])
            self.assertEqual(float(built.getPercOfOverlappingColumns()), expected[5])

            # check integration
            if self.integration:
                instance.generate(nclusters=5)

    def test_quality(self):

        datasets = ['Numeric', 'Symbolic']

        quality_params = [
            [0.12, 0.10, 0.20, 0.1, 0.0, 0.1, 0.23, 0.05],
            [0.12, 0.10, 0.20, 1.0, 0.1, 0.1, 0.23, 1.0],
        ]

        expected_params = [
            [0.12, 0.10, 0.20, 0.1, 0.0, 0.1, 0.23, 0.05],
            [0.12, 0.10, 0.20, 1, 0.1, 0.1, 0.23, 1],
        ]

        for i, (ds, params, expected) in enumerate(zip(datasets, quality_params, expected_params)):

            # build instance
            instance = bg(dstype=ds, percmissingsonbackground=params[0], percmissingsonclusters=params[1],
                          percnoiseonclusters=params[2], percnoisedeviation=params[3], percnoiseonbackground=params[4],
                          percerroesonbackground=params[5], percerrorsonclusters=params[6],
                          percerrorondeviation=params[7], silence=True)

            # check values
            self.assertEqual(instance.missing, (expected[0], expected[1]))
            self.assertEqual(instance.noise, (expected[4], expected[2], expected[3]))
            self.assertEqual(instance.errors, (expected[5], expected[6], expected[7]))

            # check integration
            if self.integration:
                instance.generate()

    def test_generate(self):

        datasets = ['Numeric', 'Numeric', 'Symbolic']

        generate_params = [
            [100, 100, 2, False, True],
            [100, 50, 2, False, False],
            [1000, 200, 5, True, None]
        ]

        expected_params = [
            [100, 100, 2, False, True],
            [100, 50, 2, False, False],
            [1000, 200, 5, True, False]
        ]

        for i, (ds, params, expected) in enumerate(zip(datasets, generate_params, expected_params)):

            # build instance
            instance = bg(dstype=ds, silence=True)

            x, y = instance.generate(
                nrows=params[0], ncols=params[1], nclusters=params[2], no_return=params[3], in_memory=params[4]
            )

            # assert dataset type
            self.assertEqual(instance.dstype, ds.upper())

            expected_shape = (expected[0], expected[1])
            gends = instance.generatedDataset
            shape = (int(gends.getNumRows()), int(gends.getNumCols()))

            if ds.upper() == 'NUMERIC':
                self.assertTrue(isinstance(gends, bic_n_dataset))

            else:
                self.assertTrue(isinstance(gends, bic_s_dataset))

            # assert gends shape
            self.assertEqual(shape, expected_shape)
            self.assertEqual(int(gends.getNumBics()), expected[2])

            # check no_return
            if expected[3]:
                self.assertIsNone(x)
                self.assertIsNone(y)

            else:
                # check to_tensor
                self.assertTrue(isinstance(y, list))
                self.assertEqual(len(y), expected[2])

                if expected[4]:
                    self.assertTrue(isinstance(x, numpy.ndarray))

                else:
                    self.assertTrue(isinstance(x, csr_matrix))

                self.assertEqual(x.shape, expected_shape)

    def test_seed(self):

        seed_params = [-1, 1, 5]

        for i, seed in enumerate(seed_params):

            # build instances

            instance_1 = bg(seed=seed, silence=True)
            x1, y1 = instance_1.generate()

            instance_2 = bg(seed=seed, silence=True)
            x2, y2 = instance_2.generate()

            # assert argument logic
            if seed != -1:
                self.assertTrue(numpy.array_equal(x1, x2))
                self.assertEqual(y1, y2)

            else:
                self.assertFalse(numpy.array_equal(x1, x2))
                self.assertNotEqual(y1, y2)

    def test_save(self):

        save_params = [
            ['default', 'example', None, None],
            ['default', 'example', os.getcwd() + '/', True],
            ['default', 'example', os.getcwd() + '/', False],
            ['csv', 'example', None, None],
            ['csv', 'example', os.getcwd() + '/', True],
            ['csv', 'example', os.getcwd() + '/', False],

        ]

        for i, params in enumerate(save_params):

            # build instance

            instance = bg(silence=True)
            instance.generate()
            instance.save(*params)

            if params[2] is None:
                params[2] = os.getcwd() + '/'

            # Assert data files were created
            if instance._asses_memory(params[3], gends=instance.generatedDataset):

                if params[0] == 'default':
                    suffix = '.tsv'
                else:
                    suffix = '.csv'
            else:

                if params[0] == 'default':
                    suffix = '_0.txt'
                else:
                    suffix = '_0.csv'

            self.assertIsFile(os.path.join(params[2], '{}_dataset{}'.format(params[1], suffix)))

            # Assert descriptive files were also created

            for kind in ['json', 'txt']:
                file = '{}_cluster_data.{}'.format(params[1], kind)
                path = os.path.join(params[2], file)

                self.assertIsFile(path)

                # Remove descriptive file
                os.remove(path)

            # Remove data files
            if suffix in ['.tsv', '.csv']:
                os.remove(os.path.join(params[2], '{}_dataset{}'.format(params[1], suffix)))
            else:
                try:
                    if params[0] == 'default':
                        for j in range(1000):
                            os.remove(os.path.join(params[2], '{}_dataset_{}.txt'.format(params[1], j)))
                    else:
                        for j in range(1000):
                            os.remove(os.path.join(params[2], '{}_dataset_{}.csv'.format(params[1], j)))
                except FileNotFoundError:
                    pass

    def test_graph(self):

        datasets = ['Numeric', 'Numeric', 'Numeric', 'Numeric',
                    'Symbolic', 'Symbolic', 'Symbolic', 'Symbolic',
                    'Numeric', 'Numeric']

        generate_params = [
            [100, 100, 2, False, True, 'networkx', 'cpu'],
            [100, 100, 2, False, True, 'networkx', 'gpu'],
            [100, 100, 2, False, True, 'dgl', 'cpu'],
            [100, 100, 2, False, True, 'dgl', 'gpu'],
            [100, 100, 2, False, True, 'networkx', 'cpu'],
            [100, 100, 2, False, True, 'networkx', 'gpu'],
            [100, 100, 2, False, True, 'dgl', 'cpu'],
            [100, 100, 2, False, True, 'dgl', 'gpu'],
            [100, 100, 2, False, False, 'dgl', 'cpu'],
            [100, 100, 2, False, False, 'networkx', 'cpu'],
        ]

        expected_params = [
            [100, 100, 2, False, True, 'networkx', 'cpu'],
            [100, 100, 2, False, True, 'networkx', 'gpu'],
            [100, 100, 2, False, True, 'dgl', 'cpu'],
            [100, 100, 2, False, True, 'dgl', 'gpu'],
            [100, 100, 2, False, True, 'networkx', 'cpu'],
            [100, 100, 2, False, True, 'networkx', 'gpu'],
            [100, 100, 2, False, True, 'dgl', 'cpu'],
            [100, 100, 2, False, True, 'dgl', 'gpu'],
            [100, 100, 2, False, False, 'dgl', 'cpu'],
            [100, 100, 2, False, False, 'networkx', 'cpu']
        ]

        for i, (ds, params, expected) in enumerate(zip(datasets, generate_params, expected_params)):

            # build instance
            instance = bg(dstype=ds, silence=True)

            instance.generate(
                nrows=params[0], ncols=params[1], nclusters=params[2], no_return=params[3], in_memory=params[4]
            )

            instance.to_graph(framework=params[5], device=params[6])

            # assert dataset type
            self.assertEqual(instance.dstype, ds.upper())

            expected_shape = (expected[0] + expected[1], expected[0] * expected[1])

            # check method logic
            if params[5] == 'networkx' and (params[6] != 'gpu' or not th.cuda.is_available()):

                if params[6] == 'gpu':
                    self.assertWarns(UserWarning)

                # assert class and define shape for networkx
                self.assertTrue(isinstance(instance.graph, Graph))
                shape = (len(instance.graph.nodes), len(instance.graph.edges))
                self.assertTrue(networkx.is_bipartite(instance.graph))

                # define 0,0 weight
                w00 = instance.graph.edges.get(('row-0', 'col-0'))['weight']

            else:

                # assert class and  define shape for dgl
                self.assertTrue(isinstance(instance.graph, dgl.DGLGraph))
                shape = (instance.graph.num_nodes(), instance.graph.num_edges())
                w00 = instance.graph.edata['w'][0]

                if params[6] == 'gpu':

                    if th.cuda.is_available():

                        if params[5] == 'networkx':
                            self.assertWarns(UserWarning)

                        self.assertEqual(instance.graph.device.type.lower(), 'gpu')

                    else:
                        self.assertWarns(UserWarning)
                        self.assertEqual(instance.graph.device.type.lower(), 'cpu')

                else:
                    self.assertEqual(instance.graph.device.type.lower(), 'cpu')

            # assert shape
            self.assertEqual(expected_shape, shape)

            # assert weight
            try:
                self.assertEqual(float(w00), float(instance.X[0][0]))
            except TypeError:
                self.assertEqual(float(w00), float(numpy.array(instance.X.todense())[0][0]))

    def test_configfile(self):

        config_params = [
            {'silence': True}
        ]

        expected_params = [
            {'silence': True}
        ]

        for i, (params, expected) in enumerate(zip(config_params, expected_params)):

            # Dump constructor params into file
            file = 'configtest.json'
            with open(file, 'w') as outfile:
                json.dump(params, outfile)

            # test construct instance
            instance = bgconfig(file)
            self.assertTrue(isinstance(instance, bgconfig))
            self.assertTrue(instance.silenced)

            # test integration
            if self.integration:
                instance.generate()
                self.assertTrue(
                    isinstance(instance.generatedDataset, bic_n_dataset) or
                    isinstance(instance.generatedDataset, bic_s_dataset)
                )

            # delete file
            os.remove(file)


class TricsGenTest(TestCaseBase):

    def setUp(self):

        self.integration = True

    def test_background(self):
        # Test initialization

        probs = [0.5, 0.25, 0.25]

        instance_numeric_uniform = tg(silence=True)
        instance_numeric_missing = tg(bktype='Missing', silence=True)
        instance_numeric_discrete = tg(bktype='Discrete', minval=1, maxval=3, probs=probs, silence=True)
        instance_numeric_normal = tg(bktype='Normal', silence=True)

        instance_symbolic_uniform = tg(dstype='SYMBOLIC', silence=True)
        instance_symbolic_missing = tg(dstype='symbolic', bktype='Missing', silence=True)
        instance_symbolic_discrete = tg(dstype='Symbolic', bktype='Discrete', nsymbols=3, probs=probs, silence=True)
        instance_symbolic_discrete_noprobs = tg(dstype='Symbolic', bktype='Discrete', nsymbols=3, silence=True)
        instance_symbolic_normal = tg(dstype='Symbolic', bktype='Normal', silence=True)

        # test initialization

        self.assertEqual(instance_numeric_uniform.background, ['UNIFORM'])
        self.assertEqual(instance_numeric_missing.background, ['MISSING'])
        self.assertEqual(instance_numeric_discrete.background, ['DISCRETE', probs])
        self.assertEqual(instance_numeric_normal.background, ['NORMAL', 14, 7])

        self.assertEqual(instance_symbolic_uniform.background, ['UNIFORM'])
        self.assertEqual(instance_symbolic_missing.background, ['MISSING'])
        self.assertEqual(instance_symbolic_discrete.background, ['DISCRETE', probs])
        self.assertEqual(instance_symbolic_discrete_noprobs.background, ['UNIFORM'])
        self.assertEqual(instance_symbolic_normal.background, ['NORMAL', 14, 7])

        # test method

        self.assertTrue(isinstance(instance_numeric_uniform._build_background(), tric_background))
        self.assertTrue(isinstance(instance_numeric_missing._build_background(), tric_background))
        self.assertTrue(isinstance(instance_numeric_discrete._build_background(), tric_background))
        self.assertTrue(isinstance(instance_numeric_normal._build_background(), tric_background))

        self.assertTrue(isinstance(instance_symbolic_uniform._build_background(), tric_background))
        self.assertTrue(isinstance(instance_symbolic_missing._build_background(), tric_background))
        self.assertTrue(isinstance(instance_symbolic_discrete._build_background(), tric_background))
        self.assertTrue(isinstance(instance_symbolic_discrete_noprobs._build_background(), tric_background))
        self.assertTrue(isinstance(instance_symbolic_normal._build_background(), tric_background))

        # integration

        if self.integration:

            instance_numeric_uniform.generate()
            instance_numeric_missing.generate()
            instance_numeric_discrete.generate()
            instance_numeric_normal.generate()

            instance_symbolic_uniform.generate()
            instance_symbolic_missing.generate()
            instance_symbolic_discrete.generate()
            instance_symbolic_discrete_noprobs.generate()
            instance_symbolic_normal.generate()

    def test_patterns(self):

        patterns = [
            ['Numeric', [['Constant', 'CONSTANT', 'MULTIPLICATIVE'], ['CONSTANT', 'NONE', 'NONE']], None],
            ['Symbolic', [['NONE', 'NONE', 'OrDeR_PRESERVING']], 'Random'],
            ['Numeric', [['CONSTANT', 'CONStANT', 'ADDITIVE'], ['CONSTANT', 'NONE', 'NONE']], None],
            ['NUMERIC', [['COnSTaNT', 'NONE', 'none']], None]
        ]

        expected_patterns = [
            ['NUMERIC', [['CONSTANT', 'CONSTANT', 'MULTIPLICATIVE'], ['CONSTANT', 'NONE', 'NONE']], None],
            ['SYMBOLIC', [['NONE', 'NONE', 'ORDERPRESERVING']], 'RANDOM'],
            ['NUMERIC', [['CONSTANT', 'CONSTANT', 'ADDITIVE'], ['CONSTANT', 'NONE', 'NONE']], None],
            ['NUMERIC', [['CONSTANT', 'NONE', 'NONE']], None]
        ]

        for i, (pattern, expected) in enumerate(zip(patterns, expected_patterns)):

            # print(i)

            instance = tg(dstype=pattern[0], patterns=pattern[1], timeprofile=pattern[2], silence=True)
            builts = instance._build_patterns()

            for built, expect in zip(builts, expected[1]):

                self.assertTrue(isinstance(built, TriclusterPattern))

                self.assertEqual(str(built.getRowsPattern().toString()).upper(), expect[0])
                self.assertEqual(str(built.getColumnsPattern().toString()).upper(), expect[1])
                self.assertEqual(str(built.getContextsPattern().toString()).upper(), expect[2])

                if expected[2]:
                    self.assertEqual(str(built.getTimeProfile().toString()).upper(), expected[2])
                else:
                    self.assertIsNone(built.getTimeProfile())

            if self.integration:
                instance.generate()

    def test_struture(self):

        distribution = [
            [[['UNIFoRM', 1.0, 4], ['UNIFORM', 1, 4], ['UNIFoRM', 1.0, 4]], 'NONE'],
            [[['Normal', 3, 1], ['Normal', 3, 1], ['UNIFORM', 1, 4]], 'CoLUMNS'],
            [[['Normal', 3, 1], ['Normal', 3, 1], ['UNIFORM', 1, 4]], 'Contexts']
        ]

        expected_distribution = [
            [[['UNIFORM', 1, 4], ['UNIFORM', 1, 4], ['UNIFORM', 1, 4]], 'NONE'],
            [[['NORMAL', 3, 1], ['NORMAL', 3, 1], ['UNIFORM', 1, 4]], 'COLUMNS'],
            [[['NORMAL', 3, 1], ['NORMAL', 3, 1], ['UNIFORM', 1, 4]], 'CONTEXTS']
        ]

        for i, (dist, expected) in enumerate(zip(distribution, expected_distribution)):
            # build instance and tester
            instance = tg(clusterdistribution=dist[0], contiguity=dist[1], silence=True)
            built = instance._build_structure()

            # check class
            self.assertTrue(isinstance(built, TriclusterStructure))

            # check row distribution
            self.assertEqual(str(built.getRowsDistribution().toString()), expected[0][0][0])
            self.assertEqual(int(built.getRowsParam1()), expected[0][0][1])
            self.assertEqual(int(built.getRowsParam2()), expected[0][0][2])

            # check cols distribution
            self.assertEqual(str(built.getColumnsDistribution().toString()), expected[0][1][0])
            self.assertEqual(int(built.getColumnsParam1()), expected[0][1][1])
            self.assertEqual(int(built.getColumnsParam2()), expected[0][1][2])

            # check contexts distribution
            self.assertEqual(str(built.getContextsDistribution().toString()), expected[0][2][0])
            self.assertEqual(int(built.getContextsParam1()), expected[0][2][1])
            self.assertEqual(int(built.getContextsParam2()), expected[0][2][2])

            # check row contiguity
            self.assertEqual(str(built.getContiguity().toString()), expected[1])

            # check integration
            if self.integration:
                instance.generate()

    def test_generator(self):

        generator_parameters = [
            {
                'dstype': 'Numeric',
                'realval': False,
                'minval': 1,
                'maxval': 5,
                'silence': True
            },
            {
                'dstype': 'Numeric',
                'realval': True,
                'minval': 1,
                'maxval': 5,
                'silence': True
            },
            {
                'dstype': 'Symbolic',
                'nsymbols': 5,
                'symmetries': 1,
                'silence': True
            }
        ]

        expected_parameters = [
            {
                'dstype': 'NUMERIC',
                'realval': False,
                'minval': 1,
                'maxval': 5,
                'silence': True
            },
            {
                'dstype': 'NUMERIC',
                'realval': True,
                'minval': 1,
                'maxval': 5,
                'silence': True
            },
            {
                'dstype': 'SYMBOLIC',
                'nsymbols': 5,
                'symmetries': 1,
                'silence': True
            }
        ]

        for i, (params, expected) in enumerate(zip(generator_parameters, expected_parameters)):

            # build instance
            instance = tg(**params)

            background = instance._build_background()
            generate_params = [100, 100, 3, 5, background]

            parameters = instance._get_dstype_vars(*generate_params)
            built = instance._build_generator(*parameters)

            self.assertTrue(isinstance(built, getattr(gen_tric, parameters[0])))

            if expected['dstype'] == 'NUMERIC':

                self.assertEqual(parameters[1][1:6], generate_params)
                self.assertEqual(parameters[1][0], expected['realval'])
                self.assertEqual(parameters[1][6], expected['minval'])
                self.assertEqual(parameters[1][7], expected['maxval'])
                self.assertIsNotNone(parameters[2])

            else:
                self.assertEqual(parameters[1][:5], generate_params)
                self.assertEqual(len(parameters[1][5]), expected['nsymbols'])
                self.assertEqual(parameters[1][6], expected['symmetries'])
                self.assertIsNotNone(parameters[2])

            # check integration
            if self.integration:
                instance.generate()

    def test_overlapping(self):

        overlapping = [
            ['NO_OVeRLaPPING', 0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [None, 0, 0, 0.0, 1.0, 1.0, 1.0],
            ['AddITIVE', 0.5, 2, 0.5, 0.75, 0.5, 1.0],
            ['MULTiPLICATIVE', 0.25, 2.0, 0.75, 1.0, 1.0, 1.0],
            ['INTERPOLED', 0.75, 3, 1, 1, 1, 1.0],
        ]

        expected_overlapping = [
            ['NO OVERLAPPING', 0.0, 0, 0.0, 1.0, 1.0, 1.0],
            ['NONE', 0.0, 0, 0.0, 1.0, 1.0, 1.0],
            ['ADDITIVE', 0.5, 2, 0.5, 0.75, 0.5, 1.0],
            ['MULTIPLICATIVE', 0.25, 2, 0.75, 1.0, 1.0, 1.0],
            ['INTERPOLED', 0.75, 3, 1.0, 1.0, 1.0, 1.0],
        ]

        for i, (overlap, expected) in enumerate(zip(overlapping, expected_overlapping)):
            # build instance and tester
            instance = tg(
                plaidcoherency=overlap[0],
                percofoverlappingclusters=overlap[1],
                maxclustsperoverlappedarea=overlap[2],
                maxpercofoverlappingelements=overlap[3],
                percofoverlappingrows=overlap[4],
                percofoverlappingcolumns=overlap[5],
                percofoverlappingcontexts=overlap[6],
                silence=True
            )

            built = instance._build_overlapping()

            # check class
            self.assertTrue(isinstance(built, overlap_tric))

            # check values

            self.assertEqual(str(built.getPlaidCoherency().toString()).upper(), expected[0])
            self.assertEqual(float(built.getPercOfOverlappingTrics()), expected[1])
            self.assertEqual(int(built.getMaxTricsPerOverlappedArea()), expected[2])
            self.assertEqual(float(built.getMaxPercOfOverlappingElements()), expected[3])
            self.assertEqual(float(built.getPercOfOverlappingRows()), expected[4])
            self.assertEqual(float(built.getPercOfOverlappingColumns()), expected[5])
            self.assertEqual(float(built.getPercOfOverlappingContexts()), expected[6])

            # check integration
            if self.integration:
                instance.generate(nclusters=5)

    def test_quality(self):

        datasets = ['Numeric', 'Symbolic']

        quality_params = [
            [0.12, 0.10, 0.20, 0.1, 0.0, 0.1, 0.23, 0.05],
            [0.12, 0.10, 0.20, 1.0, 0.1, 0.1, 0.23, 1.0],
        ]

        expected_params = [
            [0.12, 0.10, 0.20, 0.1, 0.0, 0.1, 0.23, 0.05],
            [0.12, 0.10, 0.20, 1, 0.1, 0.1, 0.23, 1],
        ]

        for i, (ds, params, expected) in enumerate(zip(datasets, quality_params, expected_params)):

            # build instance and tester
            instance = tg(dstype=ds, percmissingsonbackground=params[0], percmissingsonclusters=params[1],
                          percnoiseonclusters=params[2], percnoisedeviation=params[3], percnoiseonbackground=params[4],
                          percerroesonbackground=params[5], percerrorsonclusters=params[6],
                          percerrorondeviation=params[7], silence=True)

            # check values
            self.assertEqual(instance.missing, (expected[0], expected[1]))
            self.assertEqual(instance.noise, (expected[4], expected[2], expected[3]))
            self.assertEqual(instance.errors, (expected[5], expected[6], expected[7]))

            # check integration
            if self.integration:
                instance.generate()

    def test_generate(self):

        datasets = ['Numeric', 'Numeric', 'Symbolic']

        generate_params = [
            [100, 100, 2, 2, False, True],
            [100, 50, 3, 2, False, False],
            [1000, 200, 50, 5, True, None]
        ]

        expected_params = [
            [100, 100, 2, 2, False, True],
            [100, 50, 3, 2, False, False],
            [1000, 200, 50, 5, True, False]
        ]

        for i, (ds, params, expected) in enumerate(zip(datasets, generate_params, expected_params)):

            # build instance
            instance = tg(dstype=ds, silence=True)

            x, y = instance.generate(
                nrows=params[0],
                ncols=params[1],
                ncontexts=params[2],
                nclusters=params[3],
                no_return=params[4],
                in_memory=params[5]
            )

            # assert dataset type
            self.assertEqual(instance.dstype, ds.upper())

            expected_shape = (expected[2], expected[0], expected[1])
            gends = instance.generatedDataset
            shape = (int(gends.getNumContexts()), int(gends.getNumRows()), int(gends.getNumCols()))

            if ds.upper() == 'NUMERIC':
                self.assertTrue(isinstance(gends, tric_n_dataset))

            else:
                self.assertTrue(isinstance(gends, tric_s_dataset))

            # assert gends shape
            self.assertEqual(shape, expected_shape)
            self.assertEqual(int(gends.getNumTrics()), expected[3])

            # check no_return
            if expected[4]:
                self.assertIsNone(x)
                self.assertIsNone(y)

            else:
                # check to_tensor
                self.assertTrue(isinstance(y, list))
                self.assertEqual(len(y), expected[3])

                if expected[5]:
                    self.assertTrue(isinstance(x, numpy.ndarray))

                else:
                    self.assertTrue(isinstance(x, COO))

                self.assertEqual(x.shape, expected_shape)

    def test_seed(self):

        seed_params = [-1, 1, 5]

        for i, seed in enumerate(seed_params):

            # build instances

            instance_1 = tg(seed=seed, silence=True)
            x1, y1 = instance_1.generate()

            instance_2 = tg(seed=seed, silence=True)
            x2, y2 = instance_2.generate()

            # assert argument logic
            if seed != -1:
                self.assertTrue(numpy.array_equal(x1, x2))
                self.assertEqual(y1, y2)

            else:
                self.assertFalse(numpy.array_equal(x1, x2))
                self.assertNotEqual(y1, y2)

    def test_save(self):

        save_params = [
            ['default', 'example', None, None],
            ['default', 'example', os.getcwd() + '/', True],
            ['default', 'example', os.getcwd() + '/', False],
            ['csv', 'example', None, None],
            ['csv', 'example', os.getcwd() + '/', True],
            ['csv', 'example', os.getcwd() + '/', False],

        ]

        for i, params in enumerate(save_params):

            # build instance

            instance = tg(silence=True)
            instance.generate()
            instance.save(*params)

            if params[2] is None:
                params[2] = os.getcwd() + '/'

            # Assert data files were created
            if instance._asses_memory(params[3], gends=instance.generatedDataset):

                if params[0] == 'default':
                    suffix = '.tsv'
                else:
                    suffix = '_ctx0.csv'
            else:
                if params[0] == 'default':
                    suffix = '_0.txt'
                else:
                    suffix = '_ctx0.csv'

            self.assertIsFile(os.path.join(params[2], '{}_dataset{}'.format(params[1], suffix)))

            # Assert descriptive files were also created

            for kind in ['json', 'txt']:
                file = '{}_cluster_data.{}'.format(params[1], kind)
                path = os.path.join(params[2], file)

                self.assertIsFile(path)

                # Remove descriptive file
                os.remove(path)

            # Remove data files
            if suffix == '.tsv':
                os.remove(os.path.join(params[2], '{}_dataset{}'.format(params[1], suffix)))
            else:
                try:
                    if params[0] == 'default':
                        for j in range(1000):
                            os.remove(os.path.join(params[2], '{}_dataset_{}.txt'.format(params[1], j)))
                    else:
                        for j in range(1000):
                            os.remove(os.path.join(params[2], '{}_dataset_ctx{}.csv'.format(params[1], j)))
                except FileNotFoundError:
                    pass

    def test_graph(self):

        datasets = ['Numeric', 'Numeric', 'Numeric', 'Numeric',
                    'Symbolic', 'Symbolic', 'Symbolic', 'Symbolic',
                    'Numeric', 'Numeric']

        generate_params = [
            [100, 100, 50, 2, False, True, 'networkx', 'cpu'],
            [100, 100, 50, 2, False, True, 'networkx', 'gpu'],
            [100, 100, 50, 2, False, True, 'dgl', 'cpu'],
            [100, 100, 50, 2, False, True, 'dgl', 'gpu'],
            [100, 100, 50, 2, False, True, 'networkx', 'cpu'],
            [100, 100, 50, 2, False, True, 'networkx', 'gpu'],
            [100, 100, 50, 2, False, True, 'dgl', 'cpu'],
            [100, 100, 50, 2, False, True, 'dgl', 'gpu'],
            [100, 100, 50, 2, False, False, 'dgl', 'cpu'],
            [100, 100, 50, 2, False, False, 'networkx', 'cpu']
        ]

        expected_params = [
            [100, 100, 50, 2, False, True, 'networkx', 'cpu'],
            [100, 100, 50, 2, False, True, 'networkx', 'gpu'],
            [100, 100, 50, 2, False, True, 'dgl', 'cpu'],
            [100, 100, 50, 2, False, True, 'dgl', 'gpu'],
            [100, 100, 50, 2, False, True, 'networkx', 'cpu'],
            [100, 100, 50, 2, False, True, 'networkx', 'gpu'],
            [100, 100, 50, 2, False, True, 'dgl', 'cpu'],
            [100, 100, 50, 2, False, True, 'dgl', 'gpu'],
            [100, 100, 50, 2, False, False, 'dgl', 'cpu'],
            [100, 100, 50, 2, False, False, 'networkx', 'cpu']
        ]

        for i, (ds, params, expected) in enumerate(zip(datasets, generate_params, expected_params)):

            # build instance
            instance = tg(dstype=ds, silence=True)

            instance.generate(
                nrows=params[0],
                ncols=params[1],
                ncontexts=params[2],
                nclusters=params[3],
                no_return=params[4],
                in_memory=params[5]
            )

            instance.to_graph(framework=params[6], device=params[7])

            # assert dataset type
            self.assertEqual(instance.dstype, ds.upper())

            expected_shape = (
                expected[0] + expected[1] + expected[2],
                expected[0] * expected[1] * expected[2] * 3
            )

            # check method logic
            if params[6] == 'networkx' and (params[7] != 'gpu' or not th.cuda.is_available()):

                if params[7] == 'gpu':
                    self.assertWarns(UserWarning)

                # assert class and define shape for networkx
                self.assertTrue(isinstance(instance.graph, Graph))
                shape = (len(instance.graph.nodes), len(instance.graph.edges))

                # define 0,0,0 weight
                w00_ = instance.graph.edges.get(('row-0', 'col-0', 0))['weight']
                w0_0 = instance.graph.edges.get(('row-0', 'ctx-0', 0))['weight']
                w_00 = instance.graph.edges.get(('col-0', 'ctx-0', 0))['weight']

            else:

                # assert class and  define shape for dgl
                self.assertTrue(isinstance(instance.graph, dgl.DGLGraph))
                shape = (instance.graph.num_nodes(), instance.graph.num_edges())

                # define 0,0,0 weight
                w00_ = instance.graph.edges[('row', 'elem', 'col')].data['w'][0]
                w0_0 = instance.graph.edges[('row', 'elem', 'ctx')].data['w'][0]
                w_00 = instance.graph.edges[('col', 'elem', 'ctx')].data['w'][0]

                if params[7] == 'gpu':

                    if th.cuda.is_available():

                        if params[6] == 'networkx':
                            self.assertWarns(UserWarning)

                        self.assertEqual(instance.graph.device.type.lower(), 'gpu')

                    else:
                        self.assertWarns(UserWarning)
                        self.assertEqual(instance.graph.device.type.lower(), 'cpu')

                else:
                    self.assertEqual(instance.graph.device.type.lower(), 'cpu')

            # assert shape
            self.assertEqual(expected_shape, shape)

            # assert weight
            self.assertTrue(w00_ == w0_0 == w_00)
            w000 = w00_
            self.assertEqual(float(w000), float(instance.X[0][0][0]))

    def test_configfile(self):

        config_params = [
            {'silence': True}
        ]

        expected_params = [
            {'silence': True}
        ]

        for i, (params, expected) in enumerate(zip(config_params, expected_params)):

            # Dump constructor params into file
            file = 'configtest.json'
            with open(file, 'w') as outfile:
                json.dump(params, outfile)

            # test construct instance
            instance = tgconfig(file)
            self.assertTrue(isinstance(instance, tgconfig))
            self.assertTrue(instance.silenced)

            # test integration
            if self.integration:
                instance.generate()
                self.assertTrue(
                    isinstance(instance.generatedDataset, tric_n_dataset) or
                    isinstance(instance.generatedDataset, tric_s_dataset)
                )

            # delete file
            os.remove(file)


if __name__ == '__main__':
    unittest.main()
