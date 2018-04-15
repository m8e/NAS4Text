#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

from libs.layers.net_code import NetCodeEnum
from libs.utils.args import get_args
from libs.trainer import Trainer

__author__ = 'fyabc'


def main(args=None):
    logging.basicConfig(
        format='{levelname}:{message}',
        level=logging.INFO,
        style='{',
    )

    net_code = [
        [
            [NetCodeEnum.LSTM, 0, 1],
            [NetCodeEnum.Convolutional, 2, 1, 0],
            [NetCodeEnum.Attention, 0],  # Cause error now here
        ],
        [
            [NetCodeEnum.LSTM, 1, 0],
        ]
    ]

    hparams = get_args(args)

    trainer = Trainer(hparams, net_code)

    trainer.train()


if __name__ == '__main__':
    main(['-T', 'de_en_iwslt', '-H', 'normal'])
