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

__author__ = 'fyabc'


class NetCodeEnum:
    # Layer types.
    LSTM = 0
    Convolutional = 1
    Attention = 2

    # Recurrent hyperparameters.

    # Convolutional hyperparameters.

    # Attention hyperparameters.


def dump_json(net_code):
    pass


def load_json(fp):
    pass


def get_net_code(hparams):
    # TODO
    return [
        [
            [NetCodeEnum.LSTM, 0, 1],
            [NetCodeEnum.Convolutional, 2, 1, 0],
            [NetCodeEnum.Attention, 0],
        ],
        [
            [NetCodeEnum.LSTM, 1, 0],
        ]
    ]
