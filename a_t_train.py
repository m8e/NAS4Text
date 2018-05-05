#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Training script of Annotated Transformer."""

from libs.utils.args import get_args
from libs.annotated_transformer.train import annotated_train_main

__author__ = 'fyabc'


def main(args=None):
    hparams = get_args(args)

    annotated_train_main(hparams)


if __name__ == '__main__':
    main()
