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
        cls.TaskName = registration_name
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

    # Automatically set by registration.
    TaskName = ''

    SourceLang = None
    TargetLang = None

    SourceVocabSize = None
    TargetVocabSize = None

    SourceFiles = {
        'train': None,
        'dev': None,
        'test': None,
        'dict': None,
    }

    TargetFiles = {
        'train': None,
        'dev': None,
        'test': None,
        'dict': None,
    }

    PAD = '<pad>'
    EOS = '<eos>'
    UNK = '<unk>'

    PAD_ID = 0
    EOS_ID = 1
    UNK_ID = 2


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

    SourceVocabSize = 32010
    TargetVocabSize = 22823

    SourceFiles = {
        'train': 'train.de-en.de',
        'dev': 'dev.de-en.de',
        'test': 'test.de-en.de',
        'dict': 'dict.de-en.de',
    }

    TargetFiles = {
        'train': 'train.de-en.en',
        'dev': 'dev.de-en.en',
        'test': 'test.de-en.en',
        'dict': 'dict.de-en.en',
    }
