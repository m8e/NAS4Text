#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Generate from the child network."""

from libs.utils.args import get_generator_args
from libs.child_generator import generate_main

__author__ = 'fyabc'


def main(args=None):
    hparams = get_generator_args(args)
    generate_main(hparams)


if __name__ == '__main__':
    main()
