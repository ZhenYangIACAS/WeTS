# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqMultiModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    base_architecture,
    Embedding,
    TransformerModel,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)

from fairseq.models.transformer import Linear

from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)

from fairseq import options
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict
import os
import math
import re
from fairseq import checkpoint_utils

def upgrade_state_dict_with_xlm_weights(
    state_dict: Dict[str, Any], pretrained_xlm_checkpoint: str
) -> Dict[str, Any]:
    """
    Load XLM weights into a Transformer encoder or decoder model.

    Args:
        state_dict: state dict for either TransformerEncoder or
            TransformerDecoder
        pretrained_xlm_checkpoint: checkpoint to load XLM weights from

    Raises:
        AssertionError: If architecture (num layers, attention heads, etc.)
            does not match between the current Transformer encoder or
            decoder and the pretrained_xlm_checkpoint
    """
    if not os.path.exists(pretrained_xlm_checkpoint):
        raise IOError("Model file not found: {}".format(pretrained_xlm_checkpoint))
    else:
        print('found pretrained xlm checkpoints {}'.format(pretrained_xlm_checkpoint))

    state = torch.load(pretrained_xlm_checkpoint, map_location='cpu')
    xlm_state_dict = state["model"]
    for key in xlm_state_dict.keys():
        for search_key in ["embed_tokens", "embed_positions", "layers"]:
            if search_key in key:
                subkey = key[key.find(search_key):]
                if subkey not in state_dict:
                    print('| {} subkey not found in state_dict'.format(subkey))
                else:
                    print('| {} subkey found in state_dict'.format(subkey))
                    state_dict[subkey] = xlm_state_dict[key]
            #else:
            #    print('| {} search_key not found in xlm_state_dict'.format(search_key))
    return state_dict


@register_model('multilingual_transformer')
class MultilingualTransformerModel(FairseqMultiModel):
    """Train Transformer models for multiple language pairs simultaneously.

    Requires `--task multilingual_translation`.

    We inherit all arguments from TransformerModel and assume that all language
    pairs use a single Transformer architecture. In addition, we provide several
    options that are specific to the multilingual setting.

    Args:
        --share-encoder-embeddings: share encoder embeddings across all source languages
        --share-decoder-embeddings: share decoder embeddings across all target languages
        --share-encoders: share all encoder params (incl. embeddings) across all source languages
        --share-decoders: share all decoder params (incl. embeddings) across all target languages
    """

    def __init__(self, encoders, decoders):
        super().__init__(encoders, decoders)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # inherit all arguments from TransformerModel
        TransformerModel.add_args(parser) 
        parser.add_argument('--share-encoder-embeddings', action='store_true',
                            help='share encoder embeddings across languages')
        parser.add_argument('--share-decoder-embeddings', action='store_true',
                            help='share decoder embeddings across languages')
        parser.add_argument('--share-encoders', action='store_true',
                            help='share encoders across languages')
        parser.add_argument('--share-decoders', action='store_true',
                            help='share decoders across languages')
        parser.add_argument('--shared-emb-langs', default=None, type=str,
                            metavar='zh-yue,en-de',help='set several languages to share embeddings')
        parser.add_argument('--shared-encoder-decoder-emb', action='store_true',
                            help='shared the embeddings of the encoder and decoder')
        parser.add_argument('--shared-decoder-emb-output', action='store_true',
                            help='shared the embeddings of the decoder and output')
        parser.add_argument('--shared-encoder-layers', default='0', type=int,
                            metavar='N', help='shared several higher layers in the encoder')

        parser.add_argument('--shared-decoder-layers', default='0', type=int,
                            metavar='N', help='shared several lower layers in the decoder')

        parser.add_argument('--encoder-pretrained-checkpoints', default='', type=str,
                            metavar='zh:path,yue:path', help='the pretrained checkpoints used to initialize the encoder; It is incompatible with deep-init methods')

        parser.add_argument('--decoder-pretrained-checkpoints', default='', type=str,
                            metavar='zh:path,yue:path', help='the pretrained checkpoints used to initialize the decoder; It is incompatible with deep-init methods')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        from fairseq.tasks.multilingual_translation import MultilingualTranslationTask
        assert isinstance(task, MultilingualTranslationTask)

        # make sure all arguments are present in older models
        base_multilingual_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        args.shared_encoder_decoder_emb = options.eval_bool(args.shared_encoder_decoder_emb)
        args.shared_decoder_emb_output = options.eval_bool(args.shared_decoder_emb_output)

        src_langs = [lang_pair.split('-')[0] for lang_pair in task.model_lang_pairs]
        tgt_langs = [lang_pair.split('-')[1] for lang_pair in task.model_lang_pairs]

        if args.share_encoders:
            utils.deprecation_warning('--share-encoders is deprecated by zieenyang')
        if args.share_decoders:
            utils.deprecation_warning('--share-decoders is deprecated by zieenyang')

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        # get the shared relations 
        if args.shared_decoder_emb_output:
            args.share_decoder_input_output_embed = True

        if args.shared_encoder_decoder_emb:
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError('--share-encoder-decoder emb requires to match --decoder-embed-dim')

        share_relation_dic = dict()
        if args.shared_emb_langs is not None:
            shared_emb_lang_pairs = args.shared_emb_langs.split(',')
            for share_item in shared_emb_lang_pairs:
                share_lang_list= share_item.split('-')
                assert len(share_lang_list) > 1
                for each_lang in share_lang_list:
                    share_relation_dic[each_lang] = share_lang_list

        # get the embedding for the encoder and decoder
        encoder_emb_dic = dict()
        decoder_emb_dic = dict()

        for source_lang in src_langs:
            if source_lang not in share_relation_dic:
                encoder_emb_dic[source_lang] = build_embedding(task.dicts[source_lang],
                            args.encoder_embed_dim, args.encoder_embed_path)
            else:
                shared_langs = share_relation_dic[source_lang]
                found_relation = False
                for each_shared_lang in shared_langs:
                    if each_shared_lang in encoder_emb_dic.keys():
                        encoder_emb_dic[source_lang] = encoder_emb_dic[each_shared_lang]
                        found_relation=True
                        break
                if not found_relation:
                    encoder_emb_dic[source_lang] = build_embedding(task.dicts[source_lang],
                            args.encoder_embed_dim, args.encoder_embed_path)

        for target_lang in tgt_langs:
            if args.shared_encoder_decoder_emb and target_lang in encoder_emb_dic.keys():
                decoder_emb_dic[target_lang] = encoder_emb_dic[target_lang]
            else:
                if target_lang not in share_relation_dic:
                    decoder_emb_dic[target_lang] = build_embedding(task.dicts[target_lang],
                               args.decoder_embed_dim, args.decoder_embed_path)
                else:
                    shared_langs = share_relation_dic[target_lang]
                    found_relation = False
                    for each_shared_lang in shared_langs:
                        if each_shared_lang in decoder_emb_dic.keys():
                            decoder_emb_dic[target_lang] = decoder_emb_dic[each_shared_lang]
                            found_relation = True
                            break
                    if not found_relation:
                        decoder_emb_dic[target_lang] = build_embedding(task.dicts[target_lang],
                               args.decoder_embed_dim, args.decoder_embed_path)
        
        # pretrained_checkpoints_dict
        enc_pretrained_lang_checkpoint_dicts = dict()
        dec_pretrained_lang_checkpoint_dicts = dict()

        if args.encoder_pretrained_checkpoints:
            """
            examples:
              zh:/data/shared/zh.pt,yue:/data/shared/yue.pt
            """
            encoder_pretrained_checkpoints = args.encoder_pretrained_checkpoints.split(',')
            for encoder_checkpoint in encoder_pretrained_checkpoints:
                lang_path_list = encoder_checkpoint.strip().split(':')
                lang, path = lang_path_list[0].strip(), lang_path_list[1].strip()
                if lang in src_langs and lang not in enc_pretrained_lang_checkpoint_dicts:
                    enc_pretrained_lang_checkpoint_dicts[lang] = path

        if args.decoder_pretrained_checkpoints:
            """
            examples:
              zh:/data/shared/zh.pt,yue:/data/shared/yue.pt
            """
            decoder_pretrained_checkpoints = args.decoder_pretrained_checkpoints.split(',')
            for decoder_checkpoint in decoder_pretrained_checkpoints:
                lang_path_list = decoder_checkpoint.strip().split(':')
                lang, path = lang_path_list[0].strip(), lang_path_list[1].strip()
                if lang in tgt_langs and lang not in dec_pretrained_lang_checkpoint_dicts: \
                    dec_pretrained_lang_checkpoint_dicts[lang] = path

        # encoders/decoders for each language
        lang_encoders, lang_decoders = {}, {}
        encoder_shared_layer_list, decoder_shared_layer_list = nn.ModuleList([]), nn.ModuleList([])

        encoder_indep_layers = args.encoder_layers - args.shared_encoder_layers
        if args.shared_encoder_layers > 0:
            assert 'adaptive' not in args.init_type, 'we do not support admin-init if sharing encoder ' \
                                                     'layers between encoders'
            encoder_shared_layer_list.extend([TransformerEncoderLayer(args, (encoder_indep_layers + i + 1) ** -0.5,
                                                                      encoder_indep_layers + i) if args.deep_init
                                              else TransformerEncoderLayer(args, layer_num=encoder_indep_layers + i)
                                              for i in range(args.shared_encoder_layers)])

        if args.shared_decoder_layers > 0:
            decoder_shared_layer_list.extend([TransformerDecoderLayer(args, no_encoder_attn=False,
                                                                      initialize_scale=(i+1) ** -0.5, layer_num=i)
                                              if args.deep_init
                                              else TransformerDecoderLayer(args, no_encoder_attn=False, layer_num=i)
                                              for i in range(args.shared_decoder_layers)])

        def get_encoder(lang):
            if lang not in lang_encoders:
                assert lang in encoder_emb_dic
                encoder_embed_tokens = encoder_emb_dic[lang]
                pretrained_checkpoint = enc_pretrained_lang_checkpoint_dicts[lang] if  \
                    lang in enc_pretrained_lang_checkpoint_dicts else None
                lang_encoders[lang] = MTransformerEncoder(args, task.dicts[lang], encoder_embed_tokens,
                                                          encoder_shared_layer_list, pretrained_checkpoint)
            return lang_encoders[lang]

        def get_decoder(lang):
            if lang not in lang_decoders:
                assert lang in decoder_emb_dic
                decoder_embed_tokens = decoder_emb_dic[lang]
                pretrained_checkpoint = dec_pretrained_lang_checkpoint_dicts[lang] if \
                    lang in dec_pretrained_lang_checkpoint_dicts else None
                lang_decoders[lang] = MTransformerDecoder(args, task.dicts[lang], decoder_embed_tokens,
                                                          False, decoder_shared_layer_list, pretrained_checkpoint)
            return lang_decoders[lang]

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src, tgt in zip(task.model_lang_pairs, src_langs, tgt_langs):
            encoders[lang_pair] = get_encoder(src)
            decoders[lang_pair] = get_decoder(tgt)

        return MultilingualTransformerModel(encoders, decoders)

    def load_state_dict(self, state_dict, strict=True):
        state_dict_subset = state_dict.copy()
        for k, _ in state_dict.items():
            assert k.startswith('models.')
            lang_pair = k.split('.')[1]
            if lang_pair not in self.models:
                del state_dict_subset[k]
        super().load_state_dict(state_dict_subset, strict=strict)


class MTransformerEncoder(FairseqEncoder):
    """
    MTransformerEncoder supports:
       shared encoder layers between multi encoders
       init encoders with pre-trained checkpoints
       init enocders with depth-init methods
    """
    def __init__(self, args, dictionary, embed_tokens, shared_encoder_layers=None, pre_trained_checkpoints=None):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.num_segments = getattr(args, 'src_segment_nums', 0)

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, embed_dim, padding_idx=None)
            if self.num_segments > 0 
            else None
        )

        self.layers = nn.ModuleList([])

        self.encoder_layers = args.encoder_layers
        self.indep_enc_layers = args.encoder_layers

        if shared_encoder_layers is None:
            self.layers.extend([
                TransformerEncoderLayer(args, (i+1) ** -0.5, i) if args.deep_init
                else TransformerEncoderLayer(args, layer_num=i)
                for i in range(args.encoder_layers)
            ])
        else:
            # we only share the several higher layers in the encoder
            # todo: setting the initialize methods for shared_encoder_layers
            shared_layer_len = len(shared_encoder_layers)
            self.layers.extend([
                TransformerEncoderLayer(args, (i + 1) ** -0.5, i) if args.deep_init
                else TransformerEncoderLayer(args, layer_num=i)
                for i in range(args.encoder_layers - shared_layer_len)
            ])
            self.layers.extend(shared_encoder_layers)
            self.indep_enc_layers = args.encoder_layers - shared_layer_len

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        # activated if language model task is used
        # todo: testing
        self.vocab_size = dictionary.__len__()
        self.lm_embed_out = None
        self.lm_learned_out_bias = None
        self.lm_task_used = getattr(args, 'use_lm_task', False)
        if self.lm_task_used:
            self.lm_learned_out_bias = nn.Parameter(torch.zeros(self.vocab_size))
            self.lm_embed_out = Linear(embed_dim, self.vocab_size, initialize_scale=None)

        if pre_trained_checkpoints:
            self.reload_pretrained_checkpoints(pre_trained_checkpoints, embed_tokens)

    def reload_pretrained_checkpoints(self, pre_trained_checkpoints, embed_tokens):
        if pre_trained_checkpoints:
            pre_trained_loaded_state_dict = upgrade_state_dict_with_xlm_weights(
                state_dict = self.state_dict(),
                pretrained_xlm_checkpoint = pre_trained_checkpoints,
            )

            # we do not initialize the shared layers from the pre-trained checkpoints
            layer_pattern = re.compile(r'layers\.(\d+)\.(.*)')
            for key in list(pre_trained_loaded_state_dict):
                if re.match(layer_pattern, key):
                    layer_num = int(re.match(layer_pattern, key).group(1))
                    if layer_num > self.indep_enc_layers - 1:
                        pre_trained_loaded_state_dict.pop(key)

            # re_allocate the size of the embed_tokens.weight, since langtok will be added into the dictionary
            key_to_be_alter_size = 'embed_tokens.weight'
            if key_to_be_alter_size in pre_trained_loaded_state_dict:
                value = pre_trained_loaded_state_dict[key_to_be_alter_size]
                value_dim = value.shape[0]
                expected_dim = len(self.dictionary)
                if value_dim < expected_dim:
                    extend_num = expected_dim - value_dim
                    extend_emb = Embedding(extend_num, embed_tokens.embedding_dim, self.dictionary.pad())
                    value = torch.cat([value, extend_emb.weight], 0)
                    pre_trained_loaded_state_dict[key_to_be_alter_size] = value
                    print('extend pretrained {0} with {1} randomly'.format(key_to_be_alter_size, extend_num))
            
            if self.indep_enc_layers != self.encoder_layers:
                # parameters of the shared encoder layers are missing
                self.load_state_dict(pre_trained_loaded_state_dict, strict=False)
            else:
                self.load_state_dict(pre_trained_loaded_state_dict, strict=True)

            print('| reload the pretrained parameters for encoder from path {}'.format(pre_trained_checkpoints))

    def forward(self, src_tokens, src_lengths, segment_labels=None):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        if not encoder_padding_mask.any():
            encoder_padding_mask = None


        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        # activated the language model task
        # todo: testing
        if self.lm_task_used:
            assert self.lm_embed_out is not None
            lm_out = self.lm_embed_out(x)
            if self.lm_learned_out_bias is not None:
                lm_out += self.lm_learned_out_bias

        if self.lm_task_used:
            return {
                'encoder_out': x,  # T x B x C
                'encoder_padding_mask': encoder_padding_mask,  # B x T
                'lm_out': lm_out  # T * B * V
            }

        else:
            return {
                'encoder_out': x,  # T x B x C
                'encoder_padding_mask': encoder_padding_mask,  # B x T
            }

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out['encoder_out'] is not None:
            encoder_out['encoder_out'] = \
                encoder_out['encoder_out'].index_select(1, new_order)
        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)
        if 'lm_out' in encoder_out and encoder_out['lm_out'] is not None:
            encoder_out['lm_out'] = \
                encoder_out['lm_out'].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            # modified by zieenyang
            weight_key = '{}.embed_positions.weight'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            elif weight_key in state_dict:
                del state_dict[weight_key]

            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class MTransformerDecoder(FairseqIncrementalDecoder):
    """
    MTransformerDncoder supports:
       shared dncoder layers between multi decoders
       init decoders with pre-trained checkpoints      
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, shared_decoder_layers=None, pre_trained_checkpoints=None):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))
        
        # added by zieenyang
        self.num_segments = getattr(args, 'tgt_segment_nums', 0)

        self.dropout = args.dropout
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)  # todo: try with input_embed_dim

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None
        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        # added by zieenyang
        self.segment_embeddings  = (
            nn.Embedding(self.num_segments, embed_dim, padding_idx=None)
            if self.num_segments > 0 
            else None
        )

        self.layers = nn.ModuleList([])

        self.decoder_layers = args.decoder_layers
        self.indep_dec_layers = args.decoder_layers

        if shared_decoder_layers is None:
            self.layers.extend([
                TransformerDecoderLayer(args, no_encoder_attn, initialize_scale=(i+1) ** -0.5, layer_num=i) if args.deep_init
                else TransformerDecoderLayer(args, no_encoder_attn, layer_num=i)
                for i in range(args.decoder_layers)
            ])
        else:
            shared_layer_len = len(shared_decoder_layers)
            self.indep_dec_layers = self.decoder_layers - shared_layer_len

            self.layers.extend(shared_decoder_layers)
            self.layers.extend([
                TransformerDecoderLayer(args, no_encoder_attn, 
                     initialize_scale=(i +1 + shared_layer_len) ** -0.5,
                     layer_num=i) 
                     if args.deep_init else TransformerDecoderLayer(args, no_encoder_attn, layer_num=i)
                for i in range(args.decoder_layers - shared_layer_len)])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None

        if pre_trained_checkpoints:
            self.reload_pre_trained_checkpoints(pre_trained_checkpoints, embed_tokens)

    def reload_pre_trained_checkpoints(self, pre_trained_checkpoints, embed_tokens):

        if pre_trained_checkpoints:
            pre_trained_loaded_state_dict = upgrade_state_dict_with_xlm_weights(
                state_dict = self.state_dict(),
                pretrained_xlm_checkpoint = pre_trained_checkpoints,
            )
            self.load_state_dict(pre_trained_loaded_state_dict, strict=True)
            print('| reload the pretrained parameters for decoder from path {}:'.format(pre_trained_checkpoints))

            # we do not initialize the shared layers from the pre-trained checkpoints
            layer_pattern = re.compile(r'layers\.(\d+)\.(.*)')
            for key in list(pre_trained_loaded_state_dict):
                if re.match(layer_pattern, key):
                    layer_num = int(re.match(layer_pattern, key).group(1))
                    if layer_num > self.indep_dec_layers - 1:
                        pre_trained_loaded_state_dict.pop(key)

            # re_allocate the size of the embed_tokens.weight, since langtok will be added into the dictionary
            key_to_be_alter_size = 'embed_tokens.weight'
            if key_to_be_alter_size in pre_trained_loaded_state_dict:
                value = pre_trained_loaded_state_dict[key_to_be_alter_size]
                value_dim = value.shape[0]
                expected_dim = len(self.dictionary)
                if value_dim < expected_dim:
                    extend_num = expected_dim - value_dim
                    extend_emb = Embedding(extend_num, embed_tokens.embedding_dim, self.dictionary.pad())
                    value = torch.cat([value, extend_emb.weight], 0)
                    pre_trained_loaded_state_dict[key_to_be_alter_size] = value
                    print('extend pretrained {0} with {1} randomly'.format(key_to_be_alter_size, extend_num))
            
            if self.indep_dec_layers != self.decoder_layers:
                # parameters of the shared encoder layers are missing
                self.load_state_dict(pre_trained_loaded_state_dict, strict=False)
            else:
                self.load_state_dict(pre_trained_loaded_state_dict, strict=True)

            print('| reload the pretrained parameters from path {} to encoder'.format(pre_trained_checkpoints))
        
    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, tgt_segment_tokens=None, **unused):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(prev_output_tokens, encoder_out, incremental_state, tgt_segment_tokens)
        x = self.output_layer(x)
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out=None, incremental_state=None, tgt_segment_tokens=None, **unused):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
            # added by zieenyang
            if tgt_segment_tokens is not None:
                tgt_segment_tokens = tgt_segment_tokens[:,-1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        # added by zieenyang
        if tgt_segment_tokens is not None and self.segment_embeddings is not None:
            x += self.segment_embeddings(tgt_segment_tokens)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        attn = None

        inner_states = [x]

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'] if encoder_out is not None else None,
                encoder_out['encoder_padding_mask'] if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self.buffered_future_mask(x) if incremental_state is None else None,
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device or self._future_mask.size(0) < dim:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict

@register_model_architecture('multilingual_transformer', 'multilingual_transformer')
def base_multilingual_architecture(args):
    base_architecture(args)
    args.share_encoders = getattr(args, 'share_encoders', False)
    args.share_decoders = getattr(args, 'share_decoders', False)
    args.finetune_encoder_bert = getattr(args, 'finetune-encoder-bert', False)
    args.encoder_teacher_path = getattr(args, 'encoder_teacher_path', '')
    args.finetune_decoder_bert = getattr(args, 'finetune-decoder-bert', False)
    args.decoder_teacher_path = getattr(args, 'decoder_teacher_path', '')


@register_model_architecture('multilingual_transformer', 'multilingual_transformer_iwslt_de_en')
def multilingual_transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_multilingual_architecture(args)
