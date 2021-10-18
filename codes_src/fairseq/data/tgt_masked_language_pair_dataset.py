# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import math

from . import data_utils, FairseqDataset

def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, src_segment_id=0, tgt_segment_id=1,
):

    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def merge_two(key1, key2, left_pad, segment1_id, segment2_id, move_eos_to_beginning=False):

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            if move_eos_to_beginning:
                assert src[-1] == eos_idx
                dst[0] = eos_idx
                dst[1:] = src[:-1]
            else:
                dst.copy_(src)

        value_1 = [s[key1] for s in samples]
        value_2 = [s[key2] for s in samples]

        lens_1 = torch.LongTensor([s[key1].numel() for s in samples])
        lens_2 = torch.LongTensor([s[key2].numel() for s in samples])

        lens = lens_1 + lens_2

        slen, bs = lens.max().item(), len(lens)
        #print('slen {} bs {}'.format(slen, bs))

        values = value_1[0].new(bs, slen).fill_(pad_idx)
        #positions = torch.arange(slen)[:, None].repeat(1, bs)
        #positions = positions.transpose(0,1).to(values.device)
        #segments = value_1[0].new(bs, slen).fill_(segment1_id)

        pos_pad_idx = 0
        seg_pad_idx = 0
        positions = value_1[0].new(bs, slen).fill_(pos_pad_idx) # pad_idx = 0
        segments = value_1[0].new(bs, slen).fill_(seg_pad_idx)
        assert segment1_id != seg_pad_idx, 'bad segment id {0} for segment1_id'.format(segment1_id)
        assert segment2_id != seg_pad_idx, 'bad segment id {0} for segment1_id'.format(segment2_id)
        

        for i in range(bs):
            len1 = lens_1[i].item()
            len2 = lens_2[i].item()
            positions_1 = torch.arange(1, 1+len1).to(positions.device)
            positions_2 = torch.arange(1, 1+len2).to(positions.device)
            segments_1 = segments.new(len1).fill_(segment1_id)
            segments_2 = segments.new(len2).fill_(segment2_id)
              
            copy_tensor(positions_1, positions[i][slen-(len1+len2):(slen-len2)] if left_pad \
                                else positions[i][:len1])  
            copy_tensor(positions_2, positions[i][(slen-len2):] if left_pad \
                                else positions[i][len1:(len1+len2)])  

            copy_tensor(segments_1, segments[i][slen-(len1+len2):(slen-len2)] if left_pad \
                                else segments[i][:len1])  
            copy_tensor(segments_2, segments[i][(slen-len2):] if left_pad \
                                else segments[i][len1:(len1+len2)])  
                               
            copy_tensor(value_1[i], values[i][slen-(len1+len2):(slen-len2)] if left_pad \
                                else values[i][:len1])
            copy_tensor(value_2[i], values[i][slen-len2:] if left_pad \
                                else values[i][len1:(len1+len2)])

        return values, positions, segments

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens, src_positions, src_segments  = merge_two('source', 
                                                         'mask_source', 
                                                         left_pad = left_pad_source,
                                                         segment1_id=src_segment_id,
                                                         segment2_id=tgt_segment_id)

    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() + s['mask_source'].numel() for s in samples])

    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)

    src_tokens = src_tokens.index_select(0, sort_order)
    src_positions = src_positions.index_select(0, sort_order)
    src_segments = src_segments.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source'] + len(s['mask_source'])) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'src_segments': src_segments,
            'src_positions': src_positions,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    return batch

class TgtMaskedLanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True, remove_eos_from_source=False, append_eos_to_target=False,
        src_segment_id=1, tgt_segment_id=2, mask_idx = 3, epoch =0,
        mask_prob = 0.15, seed = 1, mask_whole_words=None, is_training =True,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()

        self.src = src
        self.tgt = tgt

        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.src_segment_id = src_segment_id
        self.tgt_segment_id = tgt_segment_id

        self.mask_prob = mask_prob
        self.seed = seed
        self.mask_whole_words = mask_whole_words
        self.is_training = is_training
        self.mask_idx = mask_idx

        self.epoch = epoch

    def __getitem__(self, index):
        tgt_item = self.tgt[index] # self.tgt never be None
        src_item = self.src[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa

        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        if self.is_training:
            tgt_list = tgt_item.tolist()
            with data_utils.numpy_seed(self.seed, self.epoch, index):
                sz = len(tgt_list)

                assert self.mask_idx not in tgt_list, \
                    'Dataset contains mask_idx (=={}), this is not expected!'.format(
                        self.mask_idx,
                     )
                
                if self.mask_whole_words is not None:
                     #todo: mask whole words
                     pass
                
                num_mask = math.ceil(self.mask_prob * sz  + np.random.rand()) # no zero

                mask_start_idx = np.random.randint(0, sz -1)

                mask_src_list = tgt_list[:mask_start_idx]
                mask_src_list.append(self.mask_idx)
                target_list = []

                if mask_start_idx + num_mask >=sz:
                    mask_src_list.append(self.src_dict.eos()) # append eos
                    target_list = tgt_list[mask_start_idx:]  # contains eos

                else:
                    mask_src_list += tgt_list[mask_start_idx + num_mask:] # contains eos
                    target_list = tgt_list[mask_start_idx : mask_start_idx + num_mask] # no eos
                    target_list.append(self.tgt_dict.eos()) # include eos

                mask_src_item = torch.LongTensor(mask_src_list)
                target_item = torch.LongTensor(target_list)
        else:
            mask_src_item = tgt_item
            target_item = None


        return {
            'id': index,
            'source': src_item,
            'mask_source': mask_src_item,
            'target': target_item,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(), 
            left_pad_source=self.left_pad_source, 
            left_pad_target=self.left_pad_target, input_feeding=self.input_feeding,
            src_segment_id = self.src_segment_id,
            tgt_segment_id = self.tgt_segment_id,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]

        return indices[np.argsort(self.src_sizes, kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
