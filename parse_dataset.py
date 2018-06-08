#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os

from libs.utils.paths import DataDir
from libs.tasks import get_task

__author__ = 'fyabc'


def scan_file(f):
    n = 0
    total_length = 0

    for line in f:
        words = line.strip().split()
        n += 1
        total_length += len(words)

    return n, total_length


def main(args=None):
    parser = argparse.ArgumentParser('Parse the dataset, get some other information.')
    parser.add_argument('-T', '--task', help='The task to parse')

    hparams = parser.parse_args(args=args)

    task = get_task(hparams.task)

    dataset_dir = os.path.join(DataDir, task.TaskName)

    for split in ('train', 'dev', 'test'):
        for is_src in (True, False):
            filename = task.get_filename(split, is_src_lang=is_src)

            with open(os.path.join(dataset_dir, filename), 'r', encoding='utf-8') as f:
                n, total_length = scan_file(f)
                print('{} {}: {} sentences, average length = {}'.format(
                    'Src' if is_src else 'Trg', split, n, total_length / n))


if __name__ == '__main__':
    main(['-T', 'de_en_iwslt_bpe2'])
