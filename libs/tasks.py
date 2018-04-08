#! /usr/bin/python
# -*- coding: utf-8 -*-

"""All text tasks."""

import re

__author__ = 'fyabc'

_first_cap_re = re.compile('(.)([A-Z][a-z]+)')
_all_cap_re = re.compile('([a-z0-9])([A-Z])')


def _camel2snake(name):
    s1 = _first_cap_re.sub(r'\1_\2', name)
    return _all_cap_re.sub(r'\1_\2', s1).lower()


_Tasks = {}


def register_task(cls_or_name):
    """

    Args:
        cls_or_name:

    Returns:

    """
    def decorator(cls, registration_name=None):
        if registration_name in _Tasks:
            raise ValueError('Name {} already exists'.format(registration_name))
        _Tasks[registration_name] = cls
        return cls

    if isinstance(cls_or_name, str):
        return lambda cls: decorator(cls, registration_name=cls_or_name)

    name = _camel2snake(cls_or_name.__name__)
    return decorator(cls_or_name, registration_name=name)


def get_task(name):
    """Get task by name.

    Args:
        name: str

    Returns:
        TextTask
    """
    return _Tasks[name]


class Languages:
    EN = 'en'
    FR = 'fr'
    DE = 'de'
    ZH = 'zh'
    RO = 'ro'


class TextTask:
    """Base class of text tasks.

    Subclasses should override its members.
    """
    SourceLang = None
    TargetLang = None

    SourceVocabSize = None
    TargetVocabSize = None

    PAD = '<PAD>'
    EOS = '<EOS>'

    PAD_ID = 0
    EOS_ID = 1


# Some common used tasks.

@register_task
class Test(TextTask):
    """A tiny task for test."""
    SourceLang = Languages.EN
    TargetLang = Languages.FR

    SourceVocabSize = 10
    TargetVocabSize = 10


@register_task
class DeEnIwslt(TextTask):
    SourceLang = Languages.DE
    TargetLang = Languages.EN

    SourceVocabSize = 32768
    TargetVocabSize = 32768
