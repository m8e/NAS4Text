#! /usr/bin/python
# -*- coding: utf-8 -*-

from .child_net_base import ChildNetBase, EncDecChildNet, ChildIncrementalDecoderBase, ChildEncoderBase
from ..tasks import get_task
from ..layers.common import *
from ..layers.build_block import build_block
from ..layers.grad_multiply import GradMultiply

__author__ = 'fyabc'


class DartsChildEncoder(ChildEncoderBase):
    pass


class DartsChildDecoder(ChildIncrementalDecoderBase):
    pass


@ChildNetBase.register_child_net
class DartsChildNet(EncDecChildNet):
    pass


class DartsNet:
    pass
