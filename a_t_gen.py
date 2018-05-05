#! /usr/bin/python
# -*- coding: utf-8 -*-

from libs.utils.args import get_generator_args
from libs.annotated_transformer.gen import annotated_gen_main

__author__ = 'fyabc'


def main(args=None):
    hparams = get_generator_args(args)

    annotated_gen_main(hparams)


if __name__ == '__main__':
    main()
