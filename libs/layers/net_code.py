#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Convert net code between different formats.

See ``NetCodeFormat.md`` for details of net code.
"""

import json
import os
import pickle
import re

from ..utils.registry_utils import camel2snake
from ..utils.search_space import GlobalSpace

__author__ = 'fyabc'


class NetCode:
    """The net code class, which contains global code and layers code."""

    # Net code types.
    Default = 'default'
    Cell = 'cell'

    def __init__(self, net_code):
        self.original_code = net_code

        if isinstance(net_code, list):
            # Compatible with old net code.
            self.global_code = {}
            self.layers_code = net_code
            self.type = self.Default
        else:
            self.global_code = net_code.get('Global', {})
            self.layers_code = net_code.get('Layers', [])
            self.type = net_code.get('Type', self.Default)

        self.check_correctness()

    def __getitem__(self, item):
        return self.layers_code[item]

    def __len__(self):
        return len(self.layers_code)

    def __eq__(self, other):
        cls = type(self)
        if not isinstance(other, cls):
            # Compatible with old net code.
            other = cls(other)
        return self.global_code == other.global_code and \
            self.layers_code == other.layers_code and \
            self.type == other.type

    def check_correctness(self):
        # TODO
        pass

    def modify_hparams(self, hparams):
        """Modify the hparams with the global net code.

        [NOTE]: The hparams priority:
            (Low)   Values defined in ``get_hparams(hparams.hparams_set)``;
                    Values defined in global net code;
            (High)  Values defined in command line.

        Args:
            hparams:

        Returns:

        """
        for name, index in self.global_code.items():
            if getattr(hparams, camel2snake(name), None) is None:
                value = getattr(GlobalSpace, name)[index]
                setattr(hparams, camel2snake(name), value)


def dump_json(net_code, fp):
    json.dump(net_code, fp)


def load_json(fp):
    """Load net code from JSON file, remove line comments."""
    return json.loads(''.join(re.sub(r'//.*\n', '\n', _line) for _line in fp))


def load_pickle(fp):
    return pickle.load(fp)


def get_net_code(hparams, modify_hparams=True):
    """Get net code from path given by hparams.

    Args:
        hparams:
        modify_hparams: Modify the hparams with the given (global) net code.

    Returns:

    """
    with open(hparams.net_code_file, 'r') as f:
        ext = os.path.splitext(f.name)[1]
        if ext == '.json':
            code = load_json(f)
        elif ext == '.pkl':
            code = load_pickle(f)
        else:
            raise ValueError('Does not support this net code file format now')

        result = NetCode(code)
        if modify_hparams:
            result.modify_hparams(hparams)
        return result
