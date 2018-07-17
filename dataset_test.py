#! /usr/bin/python
# -*- coding: utf-8 -*-

import logging

from libs.utils.args import get_args
from libs.utils.data_processing import LanguageDatasets

__author__ = 'fyabc'


def main(args=None):
    logging.basicConfig(
        format='{levelname}:{message}',
        level=logging.DEBUG,
        style='{',
    )

    hparams = get_args(args)

    datasets = LanguageDatasets(hparams)

    print(len(datasets.source_dict))
    print(len(datasets.target_dict))

    dev_loader = datasets.eval_dataloader('dev')

    from itertools import islice

    i = 1
    s = islice(dev_loader, i, i + 1)
    print(list(s)[0])


if __name__ == '__main__':
    main(['-T', 'de_en_iwslt'])
