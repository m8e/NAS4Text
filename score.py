#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os

from libs.utils import tokenizer, dictionary
from libs.tasks import get_task

try:
    import fy_bleu
except ImportError as e:
    import sys
    sys.stderr.write('ERROR: package "fy_bleu" not installed. run `python setup.py install` in fy_bleu directory\n')
    raise e

__author__ = 'fyabc'


def main():
    parser = argparse.ArgumentParser(description='Command-line script for BLEU scoring.')
    parser.add_argument('-s', '--sys', default='-', help='system output')
    parser.add_argument('-r', '--ref', required=True, help='references')
    parser.add_argument('-o', '--order', default=4, metavar='N',
                        type=int, help='consider ngrams up to this order')
    parser.add_argument('--ignore-case', action='store_true',
                        help='case-insensitive scoring')

    args = parser.parse_args()
    print(args)

    assert args.sys == '-' or os.path.exists(args.sys), \
        "System output file {} does not exist".format(args.sys)
    assert os.path.exists(args.ref), \
        "Reference file {} does not exist".format(args.ref)

    dict_ = dictionary.Dictionary(None, task=get_task('default'), mode='empty')

    def readlines(fd):
        for line in fd.readlines():
            if args.ignore_case:
                yield line.lower()
            yield line

    def score(fdsys):
        with open(args.ref) as fdref:
            scorer = fy_bleu.Scorer(dict_.pad_id, dict_.eos_id, dict_.unk_id)
            for sys_tok, ref_tok in zip(readlines(fdsys), readlines(fdref)):
                sys_tok = tokenizer.Tokenizer.tokenize(sys_tok, dict_)
                ref_tok = tokenizer.Tokenizer.tokenize(ref_tok, dict_)
                scorer.add(ref_tok, sys_tok)
            print(scorer.result_string(args.order))

    if args.sys == '-':
        score(sys.stdin)
    else:
        with open(args.sys, 'r') as f:
            score(f)


if __name__ == '__main__':
    main()
