
from biclustgen import NumericGenerator
import numpy

test = NumericGenerator()
assert isinstance(test.generate(), numpy.ndarray)
