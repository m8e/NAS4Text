#! /usr/bin/python
# -*- coding: utf-8 -*-

"""All text tasks."""

from .utils.registry_utils import camel2snake

__author__ = 'fyabc'

AllTasks = {}


def register_task(cls_or_name):
    """

    Args:
        cls_or_name:

    Returns:

    """
    def decorator(cls, registration_name=None):
        if registration_name in AllTasks:
            raise ValueError('Name {} already exists'.format(registration_name))
        AllTasks[registration_name] = cls
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
    return AllTasks[name]


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

    # Unique filename for data files.
    UniqueFilename = 'task'

    SourceLang = None
    TargetLang = None

    SourceVocabSize = None
    TargetVocabSize = None

    PAD = '<pad>'
    EOS = '</s>'
    UNK = '<unk>'

    PAD_ID = 0
    EOS_ID = 1
    UNK_ID = 2

    # Average sentence lengths of the dataset
    LengthInfo = {
        'src': {
            'train': 1,
            'dev': 2,
            'test': 1,
        },
        'trg': {
            'train': 1,
            'dev': 2,
            'test': 1,
        },
    }

    @classmethod
    def get_lang_pair(cls):
        return [cls.SourceLang, cls.TargetLang]

    @classmethod
    def get_filename(cls, split, is_src_lang):
        """Get dataset filename.

        Args:
            split: Data split. Can be 'train', 'test', 'valid' or 'dict'.
            is_src_lang: Is source language (True) or target language (False)?

        Returns:
            String filename.
            Format:
                <split-name>.<unique-name>.<src-lang>-<trg-lang>.<current-lang>

        Examples:
            'train.iwslt.de-en.de'
        """
        return '{}.{}.{}-{}.{}'.format(split, cls.UniqueFilename, cls.SourceLang, cls.TargetLang,
                                       cls.SourceLang if is_src_lang else cls.TargetLang)

    @classmethod
    def get_avg_length(cls, split, is_src_lang):
        return cls.LengthInfo['src' if is_src_lang else 'trg'][split]

    @classmethod
    def get_maxlen_a_b(cls):
        """Get the factors a & b of target length in generation.

        Get it from average length in train and dev datasets.

        trg_length = a * src_length + b
        """
        x1, x2 = cls.LengthInfo['src']['train'], cls.LengthInfo['src']['dev']
        y1, y2 = cls.LengthInfo['trg']['train'], cls.LengthInfo['trg']['dev']
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        return a, b

    @classmethod
    def get_vocab_size(cls, is_src_lang=True):
        if is_src_lang:
            return cls.SourceVocabSize
        else:
            return cls.TargetVocabSize


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

    UniqueFilename = 'iwslt'

    SourceVocabSize = 32010
    TargetVocabSize = 22823

    LengthInfo = {
        'src': {
            'train': 18.5276143642,
            'dev': 18.5530205194,
            'test': 19.6278518519,
        },
        'trg': {
            'train': 19.5001891395,
            'dev': 19.5236045344,
            'test': 20.4282962963,
        },
    }
