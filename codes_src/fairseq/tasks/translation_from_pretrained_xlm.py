# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq import options, utils
from fairseq.data import (
    Dictionary,
    LanguagePairDataset,
    RoundRobinZipDatasets,
    indexed_dataset,
)

import torch

from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks.translation import load_langpair_dataset
from fairseq.data.add_segment_id_lang_pair_dataset import AddSegmentIdLangPairDataset

from . import register_task
import os
import copy
from collections import OrderedDict
##################################
# task for cross-lingual translation model from pretrained xlm checkpoints
# Author: Zieenyang
# Data: 20190912
###################################

@register_task("translation_from_pretrained_xlm")
class TranslationFromPretrainedXLMTask(TranslationTask):
    """
    Same as TranslationTask except use the MaskedLMDictionary class so that
    we can load data that was binarized with the MaskedLMDictionary class.

    This task should be used for the entire training pipeline when we want to
    train an NMT model from a pretrained XLM checkpoint: binarizing NMT data,
    training NMT with the pretrained XLM checkpoint, and subsequent evaluation
    of that trained model.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language (only needed for inference)')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language (only needed for inference)')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', default=False, action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')

    def __init__(self, args, src_dict, tgt_dict, training):
        super().__init__(args, src_dict, tgt_dict)
        self.training = training

        # lang_pairs/source_lang/target_lang are import parameters
        self.lang_pairs = args.lang_pairs

        if training:
            args.source_lang, args.target_lang = args.lang_pairs[0].split('-')

        self.eval_lang_pairs = self.lang_pairs
        self.model_lang_pairs = self.lang_pairs

        self.langs = sorted(list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')}))
        self.lang2id = self._lang2id(self.langs)

    def _lang2id(self, languages: str):
        lang2id = {}
        for id, lang in enumerate(languages):
            lang2id[lang] = id
        return lang2id

    @classmethod
    def setup_task(cls, args, **kwargs):
        src_dict, tgt_dict, training = cls.prepare(args, **kwargs)
        # todo: support different dic for each language
        return cls(args, src_dict, tgt_dict, training)

    @classmethod
    def prepare(cls, args, **kwargs):
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning('--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning('--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        if args.lang_pairs is None:
            raise ValueError('--lang-pairs is required. List all the language pairs in the training objective.')
        args.lang_pairs = args.lang_pairs.split(',')
        sorted_langs = sorted(list({x for lang_pair in args.lang_pairs for x in lang_pair.split('-')}))
        if args.source_lang is not None or args.target_lang is not None:
            training = False
        else:
            training = True

        paths = args.data.split(':')
        src_dict = cls.load_dictionary(os.path.join(paths[0], 'dict.txt'))
        tgt_dict = src_dict
        print('| [{}] dictionary: {} types'.format('dict.txt', len(src_dict)))
        return src_dict, tgt_dict, training

    @classmethod
    def load_dictionary(cls, filename):
        """Load the masked LM dictionary from the filename

        Args:
            filename (str): the filename
        """
        return MaskedLMDictionary.load(filename)


    def add_segment_id_dataset(self, lang_pair_dataset, src_lang, tgt_lang):
        src_segment_id = self.lang2id[src_lang]
        tgt_segment_id = self.lang2id[tgt_lang]
        return AddSegmentIdLangPairDataset(lang_pair_dataset, src_segment_id, tgt_segment_id)

    def load_dataset(self, split, epoch=0, **kwargs):
        """Load a dataset split."""

        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            langpair_dataset = load_langpair_dataset(
                data_path, split, src, self.src_dict, tgt, self.tgt_dict,
                combine=True, dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )

            return self.add_segment_id_dataset(
                langpair_dataset,
                src_lang=src,
                tgt_lang=tgt,
            )

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([
                (lang_pair, language_pair_dataset(lang_pair))
                for lang_pair in self.lang_pairs
            ]),
            eval_key=None if self.training else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        lang_pair = "%s-%s" % (self.args.source_lang, self.args.target_lang)
        return RoundRobinZipDatasets(
            OrderedDict([(
                lang_pair,
                self.add_segment_id_dataset(
                    LanguagePairDataset(
                        src_tokens, src_lengths,
                        self.source_dictionary
                    ),
                    src_lang=self.args.source_lang,
                    tgt_lang=self.args.target_lang,
                ),
            )]),
            eval_key=lang_pair,
        )

    def build_model(self, args):
        def check_args():
            messages = []
            if len(set(self.args.lang_pairs).symmetric_difference(args.lang_pairs)) != 0:
                messages.append('--lang-pairs should include all the language pairs {}.'.format(args.lang_pairs))

            if len(messages) > 0:
                raise ValueError(' '.join(messages))

        # Check if task args are consistant with model args
        check_args()

        from fairseq import models
        model = models.build_model(args, self)
        return model

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
        for lang_pair in self.model_lang_pairs:
            if sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                continue
            loss, sample_size, logging_output = criterion(model, sample[lang_pair])
            if ignore_grad:
                loss *= 0
            optimizer.backward(loss)
            agg_loss += loss.detach().item()
            # TODO make summing of the sample sizes configurable
            agg_sample_size += sample_size
            agg_logging_output[lang_pair] = logging_output
        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
            for lang_pair in self.eval_lang_pairs:
                if lang_pair not in sample or sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue
                loss, sample_size, logging_output = criterion(model, sample[lang_pair])
                agg_loss += loss.data.item()
                # TODO make summing of the sample sizes configurable
                agg_sample_size += sample_size
                agg_logging_output[lang_pair] = logging_output
        return agg_loss, agg_sample_size, agg_logging_output

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            return generator.generate(
                    models,
                    sample,
                    prefix_tokens=prefix_tokens
            )

    def init_logging_output(self, sample):
        return {
            'ntokens': sum(
                sample_lang.get('ntokens', 0)
                for sample_lang in sample.values()
            ) if sample is not None else 0,
            'nsentences': sum(
                sample_lang['target'].size(0) if 'target' in sample_lang else 0
                for sample_lang in sample.values()
            ) if sample is not None else 0,
        }

    def grad_denom(self, sample_sizes, criterion):
        return criterion.__class__.grad_denom(sample_sizes)


    def aggregate_logging_outputs(self, logging_outputs, criterion, logging_output_keys=None):
        logging_output_keys = logging_output_keys or self.eval_lang_pairs
        # aggregate logging outputs for each language pair
        agg_logging_outputs = {
            key: criterion.__class__.aggregate_logging_outputs([
                logging_output.get(key, {}) for logging_output in logging_outputs
            ])
            for key in logging_output_keys
        }

        def sum_over_languages(key):
            return sum(logging_output[key] for logging_output in agg_logging_outputs.values())

        # flatten logging outputs
        flat_logging_output = {
            '{}:{}'.format(lang_pair, k): v
            for lang_pair, agg_logging_output in agg_logging_outputs.items()
            for k, v in agg_logging_output.items()
        }
        flat_logging_output['loss'] = sum_over_languages('loss')
        if any('nll_loss' in logging_output for logging_output in agg_logging_outputs.values()):
            flat_logging_output['nll_loss'] = sum_over_languages('nll_loss')
        flat_logging_output['sample_size'] = sum_over_languages('sample_size')
        flat_logging_output['nsentences'] = sum_over_languages('nsentences')
        flat_logging_output['ntokens'] = sum_over_languages('ntokens')
        return flat_logging_output

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        if len(self.datasets.values()) == 0:
            return {'%s-%s' % (self.args.source_lang, self.args.target_lang):
                    (self.args.max_source_positions, self.args.max_target_positions)}
        return OrderedDict([
            (key, (self.args.max_source_positions, self.args.max_target_positions))
            for split in self.datasets.keys()
            for key in self.datasets[split].datasets.keys()
        ])
