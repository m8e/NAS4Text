#! /usr/bin/python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages, Extension

__author__ = 'fyabc'


bleu = Extension(
    'fy_bleu.libbleu',
    sources=[
        'fy_bleu/clib/libbleu/libbleu.cpp',
        'fy_bleu/clib/libbleu/module.cpp',
    ],
    extra_compile_args=['-std=c++11'],
)

setup(
    name='fy_bleu',
    version='0.0.1',
    description='Small libbleu library used by fyabc',
    packages=find_packages(),
    ext_modules=[bleu],
)
