#! /usr/bin/python
# -*- coding: utf-8 -*-

import re

__author__ = 'fyabc'

_first_cap_re = re.compile('(.)([A-Z][a-z]+)')
_all_cap_re = re.compile('([a-z0-9])([A-Z])')


def camel2snake(name):
    s1 = _first_cap_re.sub(r'\1_\2', name)
    return _all_cap_re.sub(r'\1_\2', s1).lower()


__all__ = [
    'camel2snake',
]
