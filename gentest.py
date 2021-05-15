from nclustgen import BiclusterGenerator as bg, TriclusterGenerator as tg
import unittest

from java.lang import System
from java.io import PrintStream


class GenTest(unittest.TestCase):

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

        # System.out and PrintStream('logs') are not comparable, everytime PrintStream('logs') is set is under a
        # unique hash, to check this the initial printstream needed to be stored and even that would not ensure
        # it was the correct one
        # TODO assert by checking if something is being logged into loggs file
        self.assertEquals(System.out, PrintStream('logs'))

        instance_T.stop_silencing()
        self.assertEquals(System.out, instance_T.stdout)

        instance_T.start_silencing(silence=False)
        self.assertEquals(System.out, instance_T.stdout)

        instance_F.start_silencing()
        self.assertEquals(System.out, instance_T.stdout)

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

        # integrated test

        # TODO generate and asses output


class BicsGenTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_background(self):
        pass

    def test_patterns(self):
        pass

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


class TricsGenTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_background(self):
        pass

    def test_patterns(self):
        pass

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
