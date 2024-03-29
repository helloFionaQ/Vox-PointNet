# -*- coding: utf-8 -*-

import json
import logging
import time

from path import Path


# 度量记录器
class MetricsLogger(object):

    def __init__(self, fname, reinitialize=False):
        self.fname = Path(fname)
        self.reinitialize = reinitialize
        if self.fname.exists():
            if self.reinitialize:
                logging.warn('{} exists, deleting'.format(self.fname))
                self.fname.remove()

    def log(self, record=None, **kwargs):
        """
        Assumption: no newlines in the input.
        """
        if record is None:
            record = {}
        record.update(kwargs)
        record['_stamp'] = time.time()
        with open(self.fname, 'ab') as f:
            f.write(json.dumps(record, ensure_ascii=True)+'\n')


def read_records(fname):
    """ convenience for reading back. """
    skipped = 0
    #二进制文件读取 rb
    with open(fname, 'rb') as f:
        for line in f:
            if not line.endswith('\n'):
                skipped += 1
                continue

                #是简单的return
            yield json.loads(line.strip())
        if skipped > 0:
            logging.warn('skipped {} lines'.format(skipped))
