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

LAYER_TYPE and LAYER_HPARAMS are all integers.
They are indices of candidate lists defined by hparams.

Meanings of LAYER_HPARAMS can be seen in lstm.py, cnn.py and attention.py.

Example:
    # Assume that all search spaces are 'normal'.
    [   # Network
        [   # Encoder
            [1, 2, 1, 0, 1, 0], # Encoder layer 0
            [1, 2, 1, 0, 1, 0], # Encoder layer 1
            [1, 2, 1, 0, 1, 0], # Encoder layer 2
            [1, 2, 1, 0, 1, 0]  # Encoder layer 3
        ],
        [   # Decoder
            [1, 2, 1, 0, 1, 0], # Decoder layer 0
            [1, 2, 1, 0, 1, 0], # Decoder layer 1
            [1, 2, 1, 0, 1, 0]  # Decoder layer 2
        ]
    ]

    => For encoder layer 0:
    code = [1, 2, 1, 0, 1, 0]
    code[0] == 1: 1 means 'Convolutional'

    => Then see the layer code format and 'normal' convolutional search space: (in search_space.py)
    ```python
    # Layer code:
    # [CNN, OutChannels, KernelSize, Stride, ..., Preprocessors, Postprocessors]

    class ConvSpaceBase:
        OutChannels = [8, 16, 32, 64]
        KernelSizes = [1, 3, 5, 7]
        Strides = [1, 2, 3]

        Preprocessors = PPPSpace.Preprocessors
        Postprocessors = PPPSpace.Postprocessors
    ```

    => So,
    code[1] == 2: 2 means OutChannels[2] -> 32
    code[2] == 1: 1 means KernelSizes[1] -> 3
    code[3] == 0: 0 means Stride[0] -> 1
    code[4] == 1: 1 means Preprocessors[1] -> Dropout   (see in search_space.py)
    code[5] == 0: 0 means Postprocessors[0] -> None     (see in search_space.py)

    => So the result layer is (you can found it in net_code_examples/fairseq_d.json):
    (layer_0): EncoderConvLayer(
        (preprocessors): ModuleList(
            (0): Dropout(p=0.1)
        )
        (postprocessors): ModuleList(
        )
        (conv): Conv1d (256, 512, kernel_size=(3,), stride=(1,))
    )

# TODO: Add support of global net code (for hyperparameters).
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


class NetCode:
    """The net code class, which contains global code and layers code."""

    def __init__(self, net_code):
        if isinstance(net_code, list):
            # Compatible with old net code.
            self.global_code = {}
            self.layers_code = net_code
        else:
            self.global_code = net_code.get('global', {})
            self.layers_code = net_code.get('layers', [])

        self.check_correctness()

    def __getitem__(self, item):
        return self.layers_code[item]

    def __len__(self):
        return len(self.layers_code)

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
        for name, value in self.global_code.items():
            if getattr(hparams, name, None) is None:
                setattr(hparams, name, value)


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
