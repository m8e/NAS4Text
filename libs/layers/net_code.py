#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Convert net code between different formats.

Standard format:
[   # Network
    [   # Encoder
        [   # Layer 0
            LAYER_TYPE,
            LAYER_HPARAMS1,
            LAYER_HPARAMS2,
            ...
        ],
        [   # Layer 1
            ...
        ],
        ...
    ],
    [   # Decoder, same as encoder
        ...
    ]
]
"""

import json
import os
import pickle
import re

__author__ = 'fyabc'


class NetCodeEnum:
    # Layer types.
    LSTM = 0
    Convolutional = 1
    Attention = 2


def dump_json(net_code, fp):
    json.dump(net_code, fp)


def load_json(fp):
    """Load net code from JSON file, remove line comments."""
    return json.loads(''.join(re.sub(r'//.*\n', '\n', _line) for _line in fp))


def load_pickle(fp):
    return pickle.load(fp)


def check_correctness(code):
    pass


def get_net_code(hparams):
    # TODO: Other format, correctness check, etc.
    with open(hparams.net_code_file, 'r') as f:
        ext = os.path.splitext(f.name)[1]
        if ext == '.json':
            code = load_json(f)
        elif ext == '.pkl':
            code = load_pickle(f)
        else:
            raise ValueError('Does not support this net code file format now')

        check_correctness(code)
        return code
