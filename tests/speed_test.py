import click
import os
from nclustgen import TriclusterGenerator, BiclusterGenerator
import torch as th
import json
from tqdm import tqdm
from time import perf_counter as pc, sleep
from statistics import mean


@click.command()
@click.option('--shape', default=None, help='Shape of dataset -> (rows, cols, contexts)')
@click.option('--hidden', default=1, help='number of hidden clusters -> int')
@click.option('--grid', default=None, help='test grid -> dict('
                                           'graphNet: Bool, graphDGLcpu: bool,graphDGLgpu: bool,save: bool})')
@click.option('--i', default=10, help='number of iterations -> int')
@click.option('--output', default='print', help='How to output results -> str(json) or str(print)')
def testcli(shape, hidden, grid, i, output):

    res = speedtest(shape, hidden, grid, i)

    if output:

        if output == 'print':
            print('Speed Results:')
            for array_type in res.keys():

                print(' {} array generation:'.format(array_type.capitalize()))
                print('     Dataset Generation Speed: {}s'.format(res[array_type]['generate']))
                print('     NetworkX Graph Generation Speed: {}s'.format(res[array_type]['graphNet']))
                print('     DGL(cpu) Graph Generation Speed: {}s'.format(res[array_type]['graphDGLcpu']))
                print('     DGL(gpu) Graph Generation Speed: {}s'.format(res[array_type]['graphDGLgpu']))
                print('     Save Speed: {}s'.format(res[array_type]['save']))

        elif output == 'json':
            with open('speedtest.json', 'w') as outfile:
                json.dump(res, outfile)


def speedtest(shape=None, hidden=1, grid=None, i=10):

    if shape is None:
        shape = (100, 100, 3)

    if grid is None:
        grid = {
            'graphNet': True,
            'graphDGLcpu': True,
            'graphDGLgpu': True,
            'save': True
        }

    # assert if bics or trics
    if shape[-1] > 1:
        generator = TriclusterGenerator

    else:
        generator = BiclusterGenerator

    results = {
        'dense': {
            'generate': [],
            'graphNet': [],
            'graphDGLcpu': [],
            'graphDGLgpu': [],
            'save': []
            },

        'sparse': {
            'generate': [],
            'graphNet': [],
            'graphDGLcpu': [],
            'graphDGLgpu': [],
            'save': []
            }
    }

    pbar = tqdm(range(i))

    for iteration in pbar:

        pbar.set_description('SPEEDTEST: {}'.format(iteration))

        for boolean, array in zip([True, False], ['dense', 'sparse']):

            # Generating DS
            start = pc()
            instance = generator(in_memory=boolean, silence=True)
            _, _ = instance.generate(*shape, nclusters=hidden)
            stop = pc()

            speedgen = stop - start

            # Test grid
            if grid['graphNet']:
                # building networkx graph

                start = pc()
                _ = instance.to_graph()
                stop = pc()

                speedgraphNet = stop - start

            else:
                # Else set speed to None (for test not run)
                speedgraphNet = None

            if grid['graphDGLcpu']:
                # building dgl graph on cpu

                start = pc()
                _ = instance.to_graph(framework='dgl')
                stop = pc()

                speedgraphDGLcpu = stop - start

            else:
                # Else set speed to None (for test not run)
                speedgraphDGLcpu = None

            # test condicions to run test
            if th.cuda.is_available() and grid['graphDGLgpu']:
                # building dgl graph on gpu

                start = pc()
                _ = instance.to_graph(framework='dgl', device='gpu')
                stop = pc()

                speedgraphDGLgpu = stop - start

            else:
                speedgraphDGLgpu = None

            if grid['save']:
                # saving

                path = os.getcwd()

                start = pc()
                instance.save(single_file=boolean)
                stop = pc()

                speedsave = stop - start

                # Remove data files
                if boolean:
                    os.remove('example_dataset.tsv')
                else:
                    try:
                        for i in range(100):
                            os.remove('example_dataset_{}.txt'.format(i))
                    except FileNotFoundError:
                        pass

                # Remove descriptive files

                os.remove('example_cluster_data.txt')
                os.remove('example_cluster_data.json')

            else:
                speedsave = None

            results[array]['generate'].append(speedgen)
            results[array]['graphNet'].append(speedgraphNet)
            results[array]['graphDGLcpu'].append(speedgraphDGLcpu)
            results[array]['graphDGLgpu'].append(speedgraphDGLgpu)
            results[array]['save'].append(speedsave)

    for array in results.keys():
        for test in results[array].keys():
            try:
                results[array][test] = mean(results[array][test])
            except TypeError:
                results[array][test] = None

    return results


if __name__ == '__main__':
    testcli()