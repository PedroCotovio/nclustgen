
from biclustgen import NumericGenerator
import numpy

test = NumericGenerator()
assert isinstance(test.generate(), numpy.ndarray)
#test.save(path='/Users/pedrocotovio/Desktop/')
#test.save()
#test.save(single_file=False)
