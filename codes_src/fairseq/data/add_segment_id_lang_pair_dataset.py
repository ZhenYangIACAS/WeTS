# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from . import FairseqDataset
from typing import Optional

####################################
# add segment tokens to the samples
# Author: zieenyang
# Data: 20190912
####################################

class AddSegmentIdLangPairDataset(FairseqDataset):
    """A :class:`~fairseq.data.FairseqDataset` wrapper that transform bos on
    collated samples of language pair dataset.

    Note that the transformation is applied in :func:`collater`.

    Args:
        dataset (~fairseq.data.FairseqDataset): dataset that collates sample into
            LanguagePairDataset schema
        src_eos (int): original source end-of-sentence symbol index to be replaced
        new_src_eos (int, optional): new end-of-sentence symbol index to replace source eos symbol
        tgt_bos (int, optional): original target beginning-of-sentence symbol index to be replaced
        new_tgt_bos (int, optional): new beginning-of-sentence symbol index to replace at the
            beginning of 'prev_output_tokens'
    """

    def __init__(
        self,
        dataset: FairseqDataset,
        src_segment_id: int,
        tgt_segment_id: int,
    ):
        self.dataset = dataset
        self.src_segment_id = src_segment_id
        self.tgt_segment_id = tgt_segment_id

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        samples = self.dataset.collater(samples)

        # TODO: support different padding direction
        if self.src_segment_id is not None:
            src_segment_tokens = samples['net_input']['src_tokens'].clone().fill_(self.src_segment_id)
            samples['net_input']['src_segment_tokens'] = src_segment_tokens

        if self.tgt_segment_id is not None and 'prev_output_tokens' in samples['net_input']:
            tgt_segment_tokens = samples['net_input']['prev_output_tokens'].clone().fill_(self.tgt_segment_id)
            samples['net_input']['tgt_segment_tokens'] = tgt_segment_tokens
            
        return samples

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)
