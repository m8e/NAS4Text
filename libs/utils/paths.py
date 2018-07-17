#! /usr/bin/python
# -*- coding: utf-8 -*-

import os

__author__ = 'fyabc'

ProjectDir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
DataDir = os.path.join(ProjectDir, 'data')
ModelDir = os.path.join(ProjectDir, 'models')
NetCodeExampleDir = os.path.join(ProjectDir, 'net_code_example')
TranslatedOutputDir = os.path.join(ProjectDir, 'translated')


def get_data_path(hparams):
    """Get data save path with given hparams."""
    if hasattr(hparams, 'data_dir') and hparams.data_dir is not None:
        data_dir = hparams.data_dir
    else:
        data_dir = DataDir
    return os.path.join(
        data_dir, hparams.task,
    )


def get_model_path(hparams):
    """Get model save path with given hparams."""
    if hasattr(hparams, 'model_dir') and hparams.model_dir is not None:
        model_dir = hparams.model_dir
    else:
        model_dir = ModelDir
    return os.path.join(
        model_dir,
        hparams.task, hparams.hparams_set,
        os.path.splitext(os.path.basename(hparams.net_code_file))[0])


def get_translate_output_path(hparams):
    """Get translate output path with given hparams."""
    if hasattr(hparams, 'output_dir') and hparams.output_dir is not None:
        output_dir = hparams.output_dir
    else:
        output_dir = TranslatedOutputDir
    return os.path.join(
        output_dir,
        hparams.task, hparams.hparams_set,
        os.path.splitext(os.path.basename(hparams.net_code_file))[0])


__all__ = [
    'get_data_path',
    'get_model_path',
    'get_translate_output_path',
]
