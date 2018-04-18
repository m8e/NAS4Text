#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Train the child network."""

from libs.utils.args import get_args
from libs.child_train_sp import single_process_main
from libs.child_train_mp import multiprocessing_main

__author__ = 'fyabc'


def main(args=None):
    hparams = get_args(args)
    if hparams.distributed_port > 0 or hparams.distributed_init_method is not None:
        raise NotImplementedError('Distributed training is not implemented')
    elif hparams.distributed_world_size > 1:
        multiprocessing_main(hparams)
    else:
        single_process_main(hparams)


if __name__ == '__main__':
    main()
