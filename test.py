import click
import nclustgen
import numpy


@click.command()
@click.option('--kind', default='TriclusterGenerator', help='Type of test')
@click.option('--save', default=False, help='If save')
def test(kind, save):

    cl = getattr(nclustgen, kind)()
    cl.generate()
    #assert isinstance(cl.generate(), numpy.ndarray)

    if save:
        cl.save(path='/Users/pedrocotovio/Desktop/', single_file=True)


if __name__ == '__main__':
    test()
