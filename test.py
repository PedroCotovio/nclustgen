import click
import nclustgen
import numpy
from scipy.sparse import csr_matrix
from sparse import COO


@click.command()
@click.option('--kind', default='TriclusterGenerator', help='Type of test')
@click.option('--memory', default=False, help='If in memory')
@click.option('--save', default=False, help='If save')
@click.option('--ifreturn', default=False, help='to omit return')
def testcli(kind, save, memory, ifreturn):

    test(kind, save, memory, ifreturn)


def test(kind='TriclusterGenerator', save=False, memory=True, ifreturn=True):

    cl = getattr(nclustgen, kind)()
    x, y = cl.generate(in_memory=memory)

    assert isinstance(y, list)

    if memory:
        assert isinstance(x, numpy.ndarray)

    else:
        assert isinstance(x, csr_matrix) or isinstance(x, COO)

    if kind == 'TriclusterGenerator':

        if x.shape != (3, 100, 100):
            raise AssertionError('wrong shape ... {}'.format(x.shape))

    else:
        if x.shape != (100, 100):
            raise AssertionError('wrong shape ... {}'.format(x.shape))

    if save:
        cl.save(path='/Users/pedrocotovio/Desktop/', single_file=True)

    if ifreturn:
        return x, y


if __name__ == '__main__':
    testcli()
