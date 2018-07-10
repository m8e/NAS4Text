#! /usr/bin/python
# -*- coding: utf-8 -*-

import os
import importlib

__author__ = 'fyabc'


# automatically import any Python files in the models/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('.' + module, package=__name__)
