# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import copy
import os

import torch

from fairseq import options, utils
from fairseq.data import (
    Dictionary,
    LanguagePairDataset,
    RoundRobinZipDatasets,
    TransformEosLangPairDataset,
    indexed_dataset,
)

from fairseq import tokenizer
from fairseq.data.legacy import MaskedLMDictionary
from fairseq.data.legacy.masked_lm_dataset import MaskedLMDataset

from fairseq.models import FairseqMultiModel
from fairseq.tasks.translation import load_langpair_dataset


from . import FairseqTask, register_task

import itertools

from fairseq.data import (
    data_utils,
    TokenBlockDataset,
)

from fairseq.data.legacy.masked_lm_dataset import MaskedLMDataset


def _lang_token(lang: str):
    return '__{}__'.format(lang)


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, \
        'cannot find language token for lang {}'.format(lang)
    return idx

def _get_mlm_dataset_key(lang_pair):
    return 'mlm_' + lang_pair

def _get_mt_dataset_key(lang_pair):
    return lang_pair

@register_task('multilingual_translation')
class MultilingualTranslationTask(FairseqTask):
    """A task for training multiple translation models simultaneously.

    We iterate round-robin over batches from multiple language pairs, ordered
    according to the `--lang-pairs` argument.

    The training loop is roughly:

        for i in range(len(epoch)):
            for lang_pair in args.lang_pairs:
                batch = next_batch_for_lang_pair(lang_pair)
                loss = criterion(model_for_lang_pair(lang_pair), batch)
                loss.backward()
            optimizer.step()

    In practice, `next_batch_for_lang_pair` is abstracted in a FairseqDataset
    (e.g., `RoundRobinZipDatasets`) and `model_for_lang_pair` is a model that
    implements the `FairseqMultiModel` interface.

    During inference it is required to specify a single `--source-lang` and
    `--target-lang`, which indicates the inference langauge direction.
    `--lang-pairs`, `--encoder-langtok`, `--decoder-langtok` have to be set to
    the same value as training.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        parser.add_argument('--mlm-lang-pairs', default='', metavar='PAIRS', type=str,
                            help='comma-separated list of language pairs to be trained with mlm(in training order): en-de,de-fr. Note: not containing lang pairs with the same source language')
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
        parser.add_argument('--tokens-per-sample', default=512, type=int, metavar='N',
                            help='max number of tokens in each sample')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--encoder-langtok', default=None, type=str, choices=['src', 'tgt'],
                            metavar='SRCTGT',
                            help='replace beginning-of-sentence in source sentence with source or target '
                                 'language token. (src/tgt)')
        parser.add_argument('--decoder-langtok', action='store_true',
                            help='replace beginning-of-sentence in target sentence with target language token')
        # added by zieenyang 2019/08/09
        parser.add_argument('--dics-path', default=None, type=str,
                            metavar='zh-dic.zh.txt',
                            help='set each language with its own dictionary file')

        parser.add_argument('--mono-data', default=None, type=str,
                            help='set the path of the monolingual data which used for mlm training')

        parser.add_argument('--use-lm-task', default=False, action='store_true',
                            help='whether to use the lm task to train the model')
        # fmt: on

    def __init__(self, args, dicts, training):
        super().__init__(args)
        self.dicts = dicts
        self.training = training
        if training:
            self.lang_pairs = args.lang_pairs
            args.source_lang, args.target_lang = args.lang_pairs[0].split('-')
        else:
            self.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = self.lang_pairs
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = self.lang_pairs
        self.langs = list(dicts.keys())
        self.mlm_lang_pairs = args.mlm_lang_pairs

    @classmethod
    def setup_task(cls, args, **kwargs):
        dicts, training = cls.prepare(args, **kwargs)
        return cls(args, dicts, training)

    @classmethod
    def prepare(cls, args, **kargs):
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

        assert not bool(args.mlm_lang_pairs) ^ args.use_lm_task, 'mlm_lang_pairs and use_mlm_task shall be set at the same time'
        # todo: testing
        # ensure the same language not occurs more than twice in the source-side mlm_lang_pairs
        def lang_to_id(languages):
            lang2id = {}
            for id, lang in enumerate(languages):
                lang2id[lang] = id
            return lang2id

        if args.mlm_lang_pairs:
            args.mlm_lang_pairs = args.mlm_lang_pairs.split(',') 
       
            sorted_mlm_source_langs = sorted(list({x for lang_pair in args.mlm_lang_pairs for x in lang_pair.split('-')}))
            mlm_lang_pre = sorted_mlm_source_langs[0]
            if len(sorted_mlm_source_langs) > 1:
                for mlm_lang in sorted_mlm_source_langs[1:]:
                    assert mlm_lang != mlm_lang_pre, '{} occurs twice or more'.format(mlm_lang)
                    mlm_lang_pre = mlm_lang

            args.lang2id = lang_to_id(sorted_mlm_source_langs)
        
        if args.dics_path is not None:
            dics_path = args.dics_path
            dics_path_list = dics_path.split(',')
            assert len(dics_path_list) == len(sorted_langs), len(sorted_langs)

            dicts = OrderedDict()
            for list_item in dics_path_list:
                paths = args.data.split(':')
                assert len(paths) > 0
                lang, dic_file_name = list_item.split('-')
                dicts[lang] = cls.load_dictionary(os.path.join(paths[0], dic_file_name))

                if len(dicts) > 0:
                    assert dicts[lang].pad() == dicts[sorted_langs[0]].pad()
                    assert dicts[lang].eos() == dicts[sorted_langs[0]].eos()
                    assert dicts[lang].unk() == dicts[sorted_langs[0]].unk()
                if args.encoder_langtok is not None or args.decoder_langtok:
                    for lang_to_add in sorted_langs:
                        dicts[lang].add_symbol(_lang_token(lang_to_add))
                print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))
        else:
            dicts = OrderedDict()
            for lang in sorted_langs:
                paths = args.data.split(':')
                assert len(paths) > 0
                dicts[lang] = cls.load_dictionary(os.path.join(paths[0], 'dict.{}.txt'.format(lang)))
                if len(dicts) > 0:
                    assert dicts[lang].pad() == dicts[sorted_langs[0]].pad()
                    assert dicts[lang].eos() == dicts[sorted_langs[0]].eos()
                    assert dicts[lang].unk() == dicts[sorted_langs[0]].unk()
                if args.encoder_langtok is not None or args.decoder_langtok:
                    for lang_to_add in sorted_langs:
                        dicts[lang].add_symbol(_lang_token(lang_to_add))
                print('| [{}] dictionary: {} types'.format(lang, len(dicts[lang])))
        return dicts, training

    def get_encoder_langtok(self, src_lang, tgt_lang):
        if self.args.encoder_langtok is None:
            return self.dicts[src_lang].eos()
        if self.args.encoder_langtok == 'src':
            return _lang_token_index(self.dicts[src_lang], src_lang)
        else:
            return _lang_token_index(self.dicts[src_lang], tgt_lang)

    def get_decoder_langtok(self, tgt_lang):
        if not self.args.decoder_langtok:
            return self.dicts[tgt_lang].eos()
        return _lang_token_index(self.dicts[tgt_lang], tgt_lang)

    def alter_dataset_langtok(self, lang_pair_dataset,
                              src_eos=None, src_lang=None, tgt_eos=None, tgt_lang=None):
        if self.args.encoder_langtok is None and not self.args.decoder_langtok:
            return lang_pair_dataset

        new_src_eos = None
        if self.args.encoder_langtok is not None and src_eos is not None \
           and src_lang is not None and tgt_lang is not None:
            new_src_eos = self.get_encoder_langtok(src_lang, tgt_lang)
        else:
            src_eos = None

        new_tgt_bos = None
        if self.args.decoder_langtok and tgt_eos is not None and tgt_lang is not None:
            new_tgt_bos = self.get_decoder_langtok(tgt_lang)
        else:
            tgt_eos = None

        return TransformEosLangPairDataset(
            lang_pair_dataset,
            src_eos=src_eos,
            new_src_eos=new_src_eos,
            tgt_bos=tgt_eos,
            new_tgt_bos=new_tgt_bos,
        )

    def load_dataset(self, split, epoch=0, **kwargs):
        """Load a dataset split."""

        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        mono_paths = []
        if self.args.mono_data:
            mono_paths = self.args.mono_data.split(':')

        if self.args.use_lm_task:
            assert len(mono_paths) > 0

        def mlm_monolingual_dataset(lang_pair):
            loaded_datasets = []
            lang = lang_pair.split('-')[0].strip()

            lang_split = '{}.{}'.format(split, lang)

            mono_data_path = mono_paths[epoch % len(mono_paths)]
            for k in itertools.count():
                split_k = lang_split + (str(k) if k > 0 else '')
                path = os.path.join(mono_data_path, split_k)
                ds = data_utils.load_indexed_dataset(path, self.dicts[lang], self.args.dataset_impl)
                if ds is None:
                    if k > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                loaded_datasets.append(
                    TokenBlockDataset(
                        ds, ds.sizes, self.args.tokens_per_sample - 1,
                        pad = self.dicts[lang].pad(), eos=self.dicts[lang].eos(),
                    )
                )

            if len(loaded_datasets) == 1:
                dataset = loaded_datasets[0]
                sizes = dataset.sizes
            else:
                dastaset = ConcatDataset(loaded_datasets)
                sizes = np.concatenate([ds.sizes for ds in loaded_datasets])

            return MaskedLMDataset(
                dataset=dataset,
                sizes=sizes,
                vocab=self.dicts[lang],
                pad_idx=self.dicts[lang].pad(),
                mask_idx=self.dicts[lang].mask(),
                classif_token_idx=self.dicts[lang].eos(),
                sep_token_idx=self.dicts[lang].eos(),
                shuffle=getattr(self.args, 'shuffle', False),
                has_pairs=False,
                segment_id=self.args.lang2id[lang],
                seed=self.args.seed)
                
        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            langpair_dataset = load_langpair_dataset(
                data_path, split, src, self.dicts[src], tgt, self.dicts[tgt],
                combine=True, dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )
            return self.alter_dataset_langtok(
                langpair_dataset,
                src_eos=self.dicts[src].eos(),
                src_lang=src,
                tgt_eos=self.dicts[tgt].eos(),
                tgt_lang=tgt,
            )

        self.datasets[split] = RoundRobinZipDatasets(
            OrderedDict([
                (_get_mt_dataset_key(lang_pair), language_pair_dataset(lang_pair))
                for lang_pair in self.lang_pairs
            ] + [
                (_get_mlm_dataset_key(lang_pair), mlm_monolingual_dataset(lang_pair))
                for lang_pair in self.mlm_lang_pairs]),
            eval_key=None if self.training else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        lang_pair = "%s-%s" % (self.args.source_lang, self.args.target_lang)
        return RoundRobinZipDatasets(
            OrderedDict([(
                lang_pair,
                self.alter_dataset_langtok(
                    LanguagePairDataset(
                        src_tokens, src_lengths,
                        self.source_dictionary
                    ),
                    src_eos=self.source_dictionary.eos(),
                    src_lang=self.args.source_lang,
                    tgt_eos=self.target_dictionary.eos(), # modified by zieenyang
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
            if self.args.encoder_langtok != args.encoder_langtok:
                messages.append('--encoder-langtok should be {}.'.format(args.encoder_langtok))
            if self.args.decoder_langtok != args.decoder_langtok:
                messages.append('--decoder-langtok should {} be set.'.format("" if args.decoder_langtok else "not"))

            if len(messages) > 0:
                raise ValueError(' '.join(messages))

        # Check if task args are consistent with model args
        check_args()

        from fairseq import models
        model = models.build_model(args, self)
        if not isinstance(model, FairseqMultiModel):
            raise ValueError('MultilingualTranslationTask requires a FairseqMultiModel architecture')
        return model

    def train_step(self, sample, model, criterion, optimizer, ignore_grad=False):
        model.train()
        agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}

        def forward_backward(model, samples, sample_key, use_mlm_loss=False, weight=1.0):
            nonlocal agg_loss, agg_sample_size, agg_logging_output
            loss, sample_size, logging_output = criterion(model, samples, use_mlm_loss=use_mlm_loss)
            if ignore_grad:
                loss *= 0

            optimizer.backward(loss)
            agg_loss += loss.detach().item()
            agg_sample_size += sample_size
            agg_logging_output[sample_key] = logging_output

        for lang_pair in self.model_lang_pairs:
            sample_key = _get_mt_dataset_key(lang_pair)
            #assert 0==1, sample_key
            if sample[sample_key] is None or len(sample[sample_key]) == 0:
                continue
            forward_backward(model.models[lang_pair], sample[sample_key], sample_key, use_mlm_loss=False)

        for lang_pair in self.mlm_lang_pairs:
            sample_key = _get_mlm_dataset_key(lang_pair)
            if sample[sample_key] is None or len(sample[sample_key]) == 0:
                continue
            forward_backward(model.models[lang_pair], sample[sample_key], sample_key, use_mlm_loss=True)

        if 'adaptive-profiling' == self.args.init_type:
            exit()
        return agg_loss, agg_sample_size, agg_logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            agg_loss, agg_sample_size, agg_logging_output = 0., 0., {}
            for lang_pair in self.eval_lang_pairs:
                if lang_pair not in sample or sample[lang_pair] is None or len(sample[lang_pair]) == 0:
                    continue
                loss, sample_size, logging_output = criterion(model.models[lang_pair], sample[lang_pair])
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
                    prefix_tokens=prefix_tokens,
                    bos_token=_lang_token_index(self.target_dictionary, self.args.target_lang)
                    if self.args.decoder_langtok else self.target_dictionary.eos(),
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

    @classmethod
    def load_dictionary(cls, filename):
        return MaskedLMDictionary.load(filename)

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = MaskedLMDictionary()
        for filename in filenames:
            MaskedLMDictionary.add_file_to_dictionary(filename, d, tokenizer.tokenize_line, workers)
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d

    @property
    def source_dictionary(self):
        return self.dicts[self.args.source_lang]

    @property
    def target_dictionary(self):
        return self.dicts[self.args.target_lang]

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
