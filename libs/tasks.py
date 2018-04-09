#! /usr/bin/python
# -*- coding: utf-8 -*-

"""All text tasks."""

from .utils.registry_utils import camel2snake

__author__ = 'fyabc'

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

    name = camel2snake(cls_or_name.__name__)
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
