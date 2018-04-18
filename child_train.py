#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Train the child network."""

from libs.utils.args import get_args
from libs.child_train_sp import single_process_main

__author__ = 'fyabc'


def main(args=None):
    hparams = get_args(args)
    # TODO: Check hparams and call other main entry here.
    single_process_main(hparams)


if __name__ == '__main__':
    main()
