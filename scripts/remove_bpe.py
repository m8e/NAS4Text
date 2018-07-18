#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Remove BPE of a file."""

import argparse


def main(args=None):
    parser = argparse.ArgumentParser(description='Remove BPE of a file.')
    parser.add_argument('file', help='The file to be convert')
    parser.add_argument('-s', '--symbol', default='@@ ', help='BPE symbol, default is %(default)r')
    parser.add_argument('-e', '--encoding', default='utf-8', help='Encoding, default is %(default)r')

    args = parser.parse_args(args)

    print(args)

    old_file, new_file = args.file, args.file + '.orig'

    with open(old_file, 'r', encoding=args.encoding) as rf, \
            open(new_file, 'w', encoding=args.encoding) as wf:
        for line in rf:
            wf.write(line.replace(args.symbol, ''))

    print('Convert {!r} into {!r}.'.format(old_file, new_file))


if __name__ == '__main__':
    main()
