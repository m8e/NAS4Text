#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

__author__ = 'fyabc'

ProjectDir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DataDir = os.path.join(ProjectDir, 'data')
ModelDir = os.path.join(ProjectDir, 'models')
NetCodeExampleDir = os.path.join(ProjectDir, 'net_code_example')
TranslatedOutputDir = os.path.join(ProjectDir, 'translated')


def get_model_path(hparams):
    """Get model save path with given hparams."""
    return os.path.join(
        ModelDir,
        hparams.task, hparams.hparams_set,
        os.path.splitext(os.path.basename(hparams.net_code_file))[0])


def get_translate_output_path(hparams):
    return os.path.join(
        TranslatedOutputDir,
        hparams.task, hparams.hparams_set,
        os.path.splitext(os.path.basename(hparams.net_code_file))[0])
