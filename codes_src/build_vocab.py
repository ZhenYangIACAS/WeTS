#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

from collections import Counter
from itertools import zip_longest

from fairseq import options, tasks, utils
from fairseq.data import indexed_dataset, Dictionary
from fairseq.binarizer import Binarizer
from multiprocessing import Pool

from fairseq import tokenizer

import os
import shutil
import sys


def build_vocab(filenames, dest_path, workers=1, threshold=-1, nwords=-1, padding_factor=8):
   d = Dictionary()
   for filename in filenames:
       Dictionary.add_file_to_dictionary(filename, d, tokenizer.tokenize_line, workers)
   d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
   d.save(dest_path)

    
if __name__ == "__main__":
    filenames = sys.argv[1:-2]
    dest_path = sys.argv[-2]
    workers = int(sys.argv[-1])
    build_vocab(filenames, dest_path, workers)
