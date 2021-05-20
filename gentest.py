from nclustgen import BiclusterGenerator as bg, TriclusterGenerator as tg
import unittest
import pathlib as pl
import numpy
from scipy.sparse import csr_matrix
from sparse import COO

from java.lang import System
from java.io import PrintStream

from com.gbic.types import Background as bic_background
from com.gtric.types import Background as tric_background
from com.gbic.utils import SingleBiclusterPattern
from com.gtric.utils import TriclusterPattern


class TestCaseBase(unittest.TestCase):

    @staticmethod
    def assertIsFile(path):
        if not pl.Path(path).resolve().is_file():
            raise AssertionError("File does not exist: %s" % str(path))

    @staticmethod
    def assertIsNotFile(path):
        if pl.Path(path).resolve().is_file():
            raise AssertionError("File exists: %s" % str(path))

class GenTest(TestCaseBase):

    def test_silence(self):

        # start silenced and un-silenced objects
        instance_T = bg(silence=True)
        instance_F = bg(silence=False)

        # check objects params
        self.assertTrue(instance_T.silenced)
        self.assertFalse(instance_F.silenced)

        self.assertIsNotNone(instance_T.stdout)
        self.assertIsNotNone(instance_F.stdout)

        # enforce System default status
        System.setOut(instance_T.stdout)

        # check method logic

        instance_T.start_silencing()
        self.assertIsFile('logs')

        instance_T.stop_silencing()
        self.assertIsNotFile('logs')
        self.assertEqual(System.out, instance_T.stdout)

        instance_T.start_silencing(silence=False)
        self.assertIsNotFile('logs')
        self.assertEqual(System.out, instance_T.stdout)

        instance_F.start_silencing()
        self.assertIsNotFile('logs')
        self.assertEqual(System.out, instance_T.stdout)

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
        self.assertTrue(instance_T.asses_memory())
        self.assertTrue(instance_T.asses_memory(in_memory=True))
        self.assertFalse(instance_T.asses_memory(in_memory=False))

        self.assertTrue(instance_T.asses_memory(gends=gends_T))
        self.assertTrue(instance_T.asses_memory(gends=gends_F))

        self.assertFalse(instance_F.asses_memory())
        self.assertTrue(instance_F.asses_memory(in_memory=True))
        self.assertFalse(instance_F.asses_memory(in_memory=False))

        self.assertFalse(instance_F.asses_memory(gends=gends_T))
        self.assertFalse(instance_F.asses_memory(gends=gends_F))

        with self.assertRaises(AttributeError):
            instance_N.asses_memory()

        self.assertTrue(instance_N.asses_memory(in_memory=True))
        self.assertFalse(instance_N.asses_memory(in_memory=False))

        self.assertTrue(instance_N.asses_memory(gends=gends_T))
        self.assertFalse(instance_N.asses_memory(gends=gends_F))


class BicsGenTest(TestCaseBase):

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

        self.assertTrue(isinstance(instance_numeric_uniform.build_background(), bic_background))
        self.assertTrue(isinstance(instance_numeric_missing.build_background(), bic_background))
        self.assertTrue(isinstance(instance_numeric_discrete.build_background(), bic_background))
        self.assertTrue(isinstance(instance_numeric_normal.build_background(), bic_background))

        self.assertTrue(isinstance(instance_symbolic_uniform.build_background(), bic_background))
        self.assertTrue(isinstance(instance_symbolic_missing.build_background(), bic_background))
        self.assertTrue(isinstance(instance_symbolic_discrete.build_background(), bic_background))
        self.assertTrue(isinstance(instance_symbolic_discrete_noprobs.build_background(), bic_background))
        self.assertTrue(isinstance(instance_symbolic_normal.build_background(), bic_background))

        # integration

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

            instance = bg(dstype=pattern[0], patterns=pattern[1], timeprofile=pattern[2], silence=True)
            builts = instance.build_patterns()

            for built, expect in zip(builts, expected[1]):

                self.assertTrue(isinstance(built, SingleBiclusterPattern))

                self.assertEqual(str(built.getBiclusterType().toString()).upper(), expected[0])
                self.assertEqual(str(built.getRowsPattern().toString()).upper(), expect[0])
                self.assertEqual(str(built.getColumnsPattern().toString()).upper(), expect[1])

                if expected[2]:
                    self.assertEqual(str(built.getTimeProfile().toString()).upper(), expected[2])
                else:
                    self.assertIsNone(built.getTimeProfile())

            instance.generate()

    def test_struture(self):
        pass

    def test_generator(self):
        pass

    def test_overlapping(self):
        pass

    def test_type(self):
        pass

    def test_quality(self):
        pass

    def test_generate(self):
        pass

    def test_save(self):
        pass

    def test_graph(self):
        pass


class TricsGenTest(TestCaseBase):

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

        self.assertTrue(isinstance(instance_numeric_uniform.build_background(), tric_background))
        self.assertTrue(isinstance(instance_numeric_missing.build_background(), tric_background))
        self.assertTrue(isinstance(instance_numeric_discrete.build_background(), tric_background))
        self.assertTrue(isinstance(instance_numeric_normal.build_background(), tric_background))

        self.assertTrue(isinstance(instance_symbolic_uniform.build_background(), tric_background))
        self.assertTrue(isinstance(instance_symbolic_missing.build_background(), tric_background))
        self.assertTrue(isinstance(instance_symbolic_discrete.build_background(), tric_background))
        self.assertTrue(isinstance(instance_symbolic_discrete_noprobs.build_background(), tric_background))
        self.assertTrue(isinstance(instance_symbolic_normal.build_background(), tric_background))

        # integration

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
            builts = instance.build_patterns()

            for built, expect in zip(builts, expected[1]):

                self.assertTrue(isinstance(built, TriclusterPattern))

                self.assertEqual(str(built.getRowsPattern().toString()).upper(), expect[0])
                self.assertEqual(str(built.getColumnsPattern().toString()).upper(), expect[1])
                self.assertEqual(str(built.getContextsPattern().toString()).upper(), expect[2])

                if expected[2]:
                    self.assertEqual(str(built.getTimeProfile().toString()).upper(), expected[2])
                else:
                    self.assertIsNone(built.getTimeProfile())

            instance.generate()

    def test_struture(self):
        pass

    def test_generator(self):
        pass

    def test_overlapping(self):
        pass

    def test_type(self):
        pass

    def test_quality(self):
        pass

    def test_generate(self):
        pass

    def test_save(self):
        pass

    def test_graph(self):
        pass


if __name__ == '__main__':
    unittest.main()
