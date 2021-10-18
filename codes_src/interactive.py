# -*- coding: utf-8 -*-

#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints


Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=input, openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer

def buffered_read_seperate_input(input, buffer_size):
    """
    This function is used by such tasks which need to read two seperate inputs for decoding,
    such as input suggestion
    Warning: inputs must be two files, not supprot stdin
    """

    buffer = []
    files = input
    assert len(files) == 2, 'bad inputs for buffered read seperate input {}'.format(files)
    with open(files[0], 'r', encoding='utf8') as f_one, open(files[1], 'r', encoding='utf8') as f_two:
        for (line_one, line_two) in zip(f_one, f_two):
            line_one, line_two = line_one.strip(), line_two.strip()
            buffer.append((line_one, line_two))
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []
    if len(buffer) > 0:
        yield buffer
 

#def make_batches(lines, args, task, max_positions, encode_fn):
#    tokens = [
#        task.source_dictionary.encode_line(
#            encode_fn(src_str), add_if_not_exist=False
#        ).long()
#        for src_str in lines
#    ]
#    lengths = torch.LongTensor([t.numel() for t in tokens])
#    itr = task.get_batch_iterator(
#        dataset=task.build_dataset_for_inference(tokens, lengths),
#        max_tokens=args.max_tokens,
#        max_sentences=args.max_sentences,
#        max_positions=max_positions,
#    ).next_epoch_itr(shuffle=False)
#    for batch in itr:
#        yield Batch(
#            ids=batch['id'],
#            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
#        )

def make_batches(lines, args, task, max_positions, encode_fn):
    """
    extend by zieenyang to support seperate inputs
    Data: 2020/12/30
    """
    def encode_fn_target(x):
        return encode_fn(x)

    if isinstance(lines[0], str):
        if args.constraints:
            """
            input line assumed with the structure: ${source_line}\t${constraint1}\t${constraint2}
            """
            batch_constraints = [list() for _ in lines]
            for i, line in enumerate(lines):
                if '\t' in line:
                    lines[i], *batch_constraints[i] = line.split('\t')

            for i, constraint_list in enumerate(batch_constraints):
                batch_constraints[i] = [
                    task.target_dictionary.encode_line(
                        encode_fn_target(constraint),
                        append_eos = False,
                        add_if_not_exist = False,
                    )
                    for constraint in constraint_list
                ]
            constraint_tensor = pack_constraints(batch_constraints)
        else:
            constraint_tensor = None

        tokens = [
            task.source_dictionary.encode_line(
                encode_fn(src_str), add_if_not_exist=False
            ).long()
            for src_str in lines
        ]
        lengths = torch.LongTensor([t.numel() for t in tokens])

    elif isinstance(lines[0], tuple):
        tokens = []
        lengths = []
        constraint_tensor = None # we do not support constrained decoding for seperate inputs now
        for tuple_item in zip(*lines):
            token_item = [
                task.source_dictionary.encode_line(
                    encode_fn(src_str), add_if_not_exist=False
                ).long()
                for src_str in tuple_item
            ]
            length_item = torch.LongTensor([t.numel() for t in token_item])
            tokens.append(token_item)
            lengths.append(length_item)
    else:
        assert False, 'bad instance for make batches'

    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(
            tokens, lengths, constraints = constraint_tensor
        ),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)

    for batch in itr:
        yield batch
        
def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')

    if args.constraints:
        print("NOTE: Constrained decoding is used")

    start_id = 0
    for inputs in buffered_read(args.input, args.buffer_size) if not args.seperate_input \
                         else buffered_read_seperate_input(args.input, args.buffer_size):
        results = []
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):

            
            src_tokens = batch['net_input']['src_tokens']
            src_lengths = batch['net_input']['src_lengths']
            src_segments = None if 'src_segments' not in batch['net_input'] \
                               else batch['net_input']['src_segments']
            src_positions = None if 'src_positions' not in batch['net_input'] \
                               else batch['net_input']['src_positions']

            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if src_segments is not None:
                    src_segments = src_segments.cuda()
                if src_positions is not None:
                    src_positions = src_positions.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }

            if src_segments is not None:
                sample['net_input']['src_segments'] = src_segments
            if src_positions is not None:
                sample['net_input']['src_positions'] = src_positions

            constraints = batch.get("constraints", None)

            translations = task.inference_step(
                generator, models, sample, constraints = constraints)

            for i, (id, hypos) in enumerate(zip(batch['id'].tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                print('S-{}\t{}'.format(id, src_str))

            # Process top predictions
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )
                hypo_str = decode_fn(hypo_str)
                print('H-{}\t{}\t{}'.format(id, hypo['score'], hypo_str))
                print('P-{}\t{}'.format(
                    id,
                    ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                ))
                if args.print_alignment:
                    print('A-{}\t{}'.format(
                        id,
                        ' '.join(map(lambda x: str(utils.item(x)), alignment))
                    ))

        # update running id counter
        start_id += len(inputs)


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
