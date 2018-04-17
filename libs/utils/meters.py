#! /usr/bin/python
# -*- coding: utf-8 -*-

import time

__author__ = 'fyabc'


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TimeMeter(object):
    """Computes the average occurrence of some event per second"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()
        self.n = 0

    def update(self, val=1):
        self.n += val

    @property
    def avg(self):
        delta = time.time() - self.start
        return self.n / delta

    @property
    def elapsed_time(self):
        return time.time() - self.start


class StopwatchMeter(object):
    """Computes the sum/avg duration of some event in seconds"""
    def __init__(self):
        self.reset()

    def start(self):
        self.start_time = time.time()

    def stop(self, n=1):
        if self.start_time is not None:
            delta = time.time() - self.start_time
            self.sum += delta
            self.n += n
            self.start_time = None

    def reset(self):
        self.sum = 0
        self.n = 0
        self.start_time = None

    @property
    def avg(self):
        return self.sum / self.n
