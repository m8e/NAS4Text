#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Define the search space here.

Layer code format can be seen in each base search space.
"""

__author__ = 'fyabc'


class GlobalSpace:
    """Global hyperparameters search space."""
    Dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    PPPDropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    AttentionDropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    FFNDropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    ClipNorm = [0.1, 0.5, 1.0, 2.5, 10, 25]


class PPPSpace:
    """Pre- and post- processors search space."""
    Dropout = 'dropout'
    Norm = 'norm'
    Residual = 'residual'

    Abbreviations = {
        'd': Dropout,
        'n': Norm,
        'a': Residual,
    }

    # None, Dropout, Norm, Dropout + Norm, Norm + Dropout, Residual, Dropout + Residual,
    # Dropout + Residual + Norm
    # TODO: Add more
    Preprocessors = [0, 1, 2, 3, 4, 5, 6, 7]
    Postprocessors = [0, 1, 2, 3, 4, 5, 6, 7]

    @classmethod
    def get_ops(cls, code):
        if isinstance(code, str):
            # "nda" -> Norm + Dropout + Residual
            return [cls.Abbreviations[c] for c in code]
        if code == 0:
            return []
        elif code == 1:
            return [cls.Dropout]
        elif code == 2:
            return [cls.Norm]
        elif code == 3:
            return [cls.Dropout, cls.Norm]
        elif code == 4:
            return [cls.Norm, cls.Dropout]
        elif code == 5:
            return [cls.Residual]
        elif code == 6:
            return [cls.Dropout, cls.Residual]
        elif code == 7:
            return [cls.Dropout, cls.Residual, cls.Norm]
        elif code is None:
            return []
        else:
            raise ValueError('Unknown code {}'.format(code))


class LayerTypes:
    """Layer types."""
    LSTM = 0
    Convolutional = 1
    Attention = 2


# LSTM.

class LSTMSpaceBase:
    """Search space of LSTM.

    Contains candidate values of hyperparameters.

    Layer code:
    [LSTM, hidden_size, Bidirectional?, ..., Preprocessors, Postprocessors]
    """

    HiddenSizes = [32, 64, 128, 256]
    UseBidirectional = [False, True]
    NumLayers = 1

    Preprocessors = PPPSpace.Preprocessors
    Postprocessors = PPPSpace.Postprocessors


class LSTMSpaceLarge(LSTMSpaceBase):
    HiddenSizes = [64, 128, 256, 512]


LSTMSpaces = {
    'base': LSTMSpaceBase,
    'large': LSTMSpaceLarge,
}


# Convolutional.

class ConvSpaceBase:
    """Search space of convolutional layers.

    Layer code:
    [CNN, OutChannels, KernelSize, Stride, ..., Preprocessors, Postprocessors]
    """

    OutChannels = [8, 16, 32, 64]
    KernelSizes = [1, 3, 5, 7]
    Strides = [1, 2, 3]
    Groups = [1, 2, 4, 8]

    Preprocessors = PPPSpace.Preprocessors
    Postprocessors = PPPSpace.Postprocessors


class ConvSpaceLarge(ConvSpaceBase):
    OutChannels = [64, 128, 256, 512]


ConvolutionalSpaces = {
    'base': ConvSpaceBase,
    'large': ConvSpaceLarge,
}


# Attention.

class AttentionSpaceBase:
    """
    Layer code:
    [Attention, NumHeads, ..., Preprocessors, Postprocessors]

    # TODO: GlobalAttention?, WindowSize, ...
    """

    # TODO: How to ensure the assertion "input_hidden_size % num_heads == 0" to be always True?
    NumHeads = [2, 4, 8, 16]
    Preprocessors = PPPSpace.Preprocessors
    Postprocessors = PPPSpace.Postprocessors
    FFNSize = [256, 512, 768, 1024, 1536, 2048]


class AttentionSpaceLarge(AttentionSpaceBase):
    pass


AttentionSpaces = {
    'base': AttentionSpaceBase,
    'large': AttentionSpaceLarge,
}


class CellSpace:
    """Search spaces of block cell."""
    CellOps = {
        k: i for i, k in enumerate(
            ['LSTM', 'CNN', 'SelfAttention', 'FFN', 'PFFN', 'Identity', 'GroupedLSTM', 'EncoderAttention', 'Zero'])
    }
    CellOpsReversed = {i: k for k, i in CellOps.items()}

    Activations = {
        k: i for i, k in enumerate(['identity', 'tanh', 'relu', 'sigmoid'])
    }
    ActivationsReversed = {i: k for k, i in Activations.items()}

    # Search space of hidden sizes. (Useless or not?)
    HiddenSizes = [32, 64, 128, 256, 512, 1024]

    CombineOps = {
        k: i for i, k in enumerate(['Add', 'Concat'])
    }
    CombineOpsReversed = {i: k for k, i in CombineOps.items()}


class DartsSpace:
    """Search spaces of DARTS and default op args."""

    # TODO: Support same op with different args
    AllCellOps = [
        ('Zero', []),
        ('Identity', []),
        ('FFN', ['relu', True]),
        ('PFFN', ['', 'd']),
        ('SelfAttention', [1, '', 'd']),
        ('EncoderAttention', [1, '', 'd']),
        ('CNN', [None, 1, 0, 0, '', 'd']),
        ('LSTM', [None, False, '', 'd']),
    ]
    DefaultCellOps = [
        ('Zero', []),
        ('Identity', []),
        ('FFN', ['relu', True]),
        ('PFFN', ['', 'd']),
        ('SelfAttention', [1, '', 'd']),
        ('EncoderAttention', [1, '', 'd']),
        ('CNN', [None, 1, 0, 0, '', 'd']),
        ('LSTM', [None, False, '', 'd']),
    ]

    CellOpSpaces = {
        'all': AllCellOps,
        'default': DefaultCellOps,
        'only-attn': [
            ('Zero', []),
            ('Identity', []),
            ('PFFN', ['', 'd', 0]),
            ('PFFN', ['', 'd', 1]),                 # Default (512) in fairseq
            ('PFFN', ['', 'd', 2]),
            ('PFFN', ['', 'd', 3]),
            ('SelfAttention', [1, '', 'd']),        # Default (4) in fairseq
            ('SelfAttention', [2, '', 'd']),
            ('EncoderAttention', [1, '', 'd']),     # Default (4) in fairseq
            ('EncoderAttention', [2, '', 'd']),
        ],
    }
    CellOpSpaces['with-r-lstm'] = CellOpSpaces['default'] + [('LSTM', [None, True, '', 'd'])]
    CellOpSpaces['no-conv-lstm'] = [o for o in CellOpSpaces['default'] if o[0] not in ('CNN', 'LSTM')]
    CellOpSpaces['no-lstm'] = [o for o in CellOpSpaces['default'] if o[0] not in ('LSTM',)]

    CombineOps = ['Add', 'Concat']
