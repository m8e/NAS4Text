# Net Code Format

1.  Standard format:

```python
{
    "Global": {     # Global net code.
        "Key 1": GLOBAL_HPARAMS1,
        "Key 2": GLOBAL_HPARAMS2,
        ...
    },
    "Layers": [     # Network
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
}
```

GLOBAL_HPARAMS, LAYER_TYPE and LAYER_HPARAMS are all integers.
They are **indices** of candidate lists defined by hparams.

Meanings of LAYER_HPARAMS can be seen in [lstm.py](libs/layers/lstm.py), [cnn.py](libs/layers/cnn.py) and [attention.py](libs/layers/attention.py).

Example:
```python
{
    "Global": {
        "Dropout": 1,
        "AttentionDropout": 1,
    }
    # Assume that all search spaces are 'normal'.
    "Layers": [     # Network
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
}
```


=> For global: (in search_space.py)

    Dropout candidates: [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    "Dropout" == 1: 1 means dropout set to 0.1

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

=> So the result layer is (you can found it in [net_code_example/fairseq_d.json](net_code_example/fairseq_d.json)):

    (layer_0): EncoderConvLayer(
        (preprocessors): ModuleList(
            (0): Dropout(p=0.1)
        )
        (postprocessors): ModuleList(
        )
        (conv): Conv1d (256, 512, kernel_size=(3,), stride=(1,))
    )

2.  Block-based format:
```python
{
    "Type": "BlockChildNet",    # Set child net type

    "Global": {     # Same as default.
        ...
    },
    "Layers": [     # Network, same as default.
        [   # Encoder
            [   # Layer 0
                [               # Node 0
                    None,       # Input 0 id (null if this is an input node)
                    None,       # Input 1 id (null if this is an input node)
                    None,       # Op 0 id (null if this is an input node)
                    None,       # Op 1 id (null if this is an input node)
                    None        # Combine op id (null if this is an input node)
                ],
                [ ... ],        # Node 1
                [               # Node 2
                    0,          # Input 0 id (null if this is an input node)
                    1,          # Input 1 id (null if this is an input node)
                    # TODO: Change CELL_OP to CELL_OP_LIST (contains extra arguments)?
                    CELL_OP,    # Op 0 id (null if this is an input node)
                    CELL_OP,    # Op 1 id (null if this is an input node)
                    COMBINE_OP  # Combine op id (null if this is an input node)
                ],
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
}
```
