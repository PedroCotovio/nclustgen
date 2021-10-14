
from setuptools import setup, find_packages
import os, fnmatch, re

version = ''
license = ''
with open('nclustgen/__init__.py', 'r') as fd:
    content = fd.read()
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE).group(1)

    license = re.search(
        r'^__license__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE).group(1)

if version is None:
    raise RuntimeError('Cannot find version information')

if license is None:
    raise RuntimeError('Cannot find license information')

def find_files(package_name,directory, pattern):
    for root, dirs, files in os.walk(os.path.join(package_name, directory)):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root[len(package_name)+1:], basename)
                yield filename

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nclustgen",
    version=version,
    author="Pedro Cotovio",
    author_email="pgcotovio@gmail.com",
    license=license,
    description="Tool to generate biclustering and triclustering datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PedroCotovio/nclustgen",
    project_urls={
        "Bug Tracker": "https://github.com/PedroCotovio/nclustgen/issues",
        "Documentation": "https://nclustgen.readthedocs.org",
    },
    install_requires=[
            'dgl>=0.6.1',
            'JPype1>=1.2.1',
            'networkx>=2.5.1',
            'torch>=1.8.1',
            'numpy',
            'scipy',
            'sparse',
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.7",

        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',


        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',

        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    include_package_data=True,
    packages=find_packages(),
    package_data={'nclustgen': list(find_files('nclustgen', 'jars/', '*.*'))},
    python_requires=">=3.7",
    keywords='biclustring triclustering generator data nclustgen',
    test_suite='tests',
)
