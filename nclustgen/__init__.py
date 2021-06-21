#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__      = "Pedro Cotovio"
__license__     = 'GNU GPLv3'
__version__ = "1.0.2"

import os
import jpype
import jpype.imports

dir_path = os.path.dirname(os.path.realpath(__file__))

# Start JVM
if jpype.isJVMStarted():
    pass
else:
    jpype.startJVM(classpath=[os.path.join(dir_path, 'jars/*')])

from .BiclusterGen import BiclusterGenerator
from .BiclusterGen import BiclusterGeneratorbyConfig
from .TriclusterGen import TriclusterGenerator
from .TriclusterGen import TriclusterGeneratorbyConfig
