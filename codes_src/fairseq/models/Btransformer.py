# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#


from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoderDecoderModel,
)

from fairseq.models.transformer import (
    base_architecture,
    Embedding,
    TransformerEncoder,
    TransformerDecoder,
    TransformerModel,
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
)

import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict
import os
from fairseq import checkpoint_utils
from fairseq.models.masked_lm_bert import MaskedLMBertModel

import torch


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

    # state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_xlm_checkpoint)
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


class BTransformerEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens, pretrained_path=None):
        super().__init__(args, dictionary, embed_tokens)
        self.mask_idx = dictionary.mask_index

        # reload the pretrained encoder parameters if provided
        # added by zieenyang at 20191009
        if pretrained_path:
            pretrained_loaded_state_dict = upgrade_state_dict_with_xlm_weights(
                state_dict = self.state_dict(),
                pretrained_xlm_checkpoint=pretrained_path,
            )
            self.load_state_dict(pretrained_loaded_state_dict, strict=True)
            print('| reload the pretrained parameters from path {} for encoder'.format(pretrained_path))

    def forward(self, src_tokens, src_lengths):
        x = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx) | src_tokens.eq(self.mask_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        if self.layer_norm:
            x = self.layer_norm(x)

        return {
            'encoder_out': x,  # T x B x C
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }


class BTransformerDecoder(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, pretrained_path=None):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if pretrained_path:
            pretrained_load_state_dict = upgrade_state_dict_with_xlm_weights(
                state_dict = self.state_dict(),
                pretrained_xlm_checkpoint=pretrained_path,
            )
            self.load_state_dict(pretrained_load_state_dict, strict=True)
            print('| reload the pretrained parameters from path {} for decoder'.format(pretrained_path))

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None, positions=None):
        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
            positions=positions,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions
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

        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = F.linear(x, self.embed_out)

        return x, {'attn': attn, 'inner_states': inner_states}


@register_model('Btransformer')
class BTransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder, teacher_encoder=None, teacher_decoder=None):
        super().__init__(encoder, decoder)
        self.args = args
        self.teacher_encoder = teacher_encoder
        self.teacher_decoder = teacher_decoder

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--encoder-pretrained-checkpoints', type=str, default=None, metavar='STR',
                            help='path for reload the pretrained encoder checkpoints')
        parser.add_argument('--decoder-pretrained-checkpoints', type=str, default=None, metavar='STR',
                            help='path for reload the pretrained decoder checkpoints')
        parser.add_argument('--encoder-teacher-path', type=str, default=None, metavar='STR',
                            help='path for reload the pretrained encoder teacher')
        parser.add_argument('--decoder-teacher-path', type=str, default=None, metavar='STR',
                            help='path for reload the pretrained decoder teacher')
        parser.add_argument('--encoder-bert-output-layer', type=int, default=-1, metavar='INT',
                            help='which layer the encoder bert outputs')
        parser.add_argument('--decoder-bert-output-layer', type=int, default=-1, metavar='INT',
                            help='which layer the decoder bert outputs')

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for the BTransformer.
        Different from the classic Transformer which only runs the forward for the encoder and decoder,
        BTransformer only runs the forward for the teacher_encoder and teacher_decoder if exits.
        """
        encoder_out = self.encoder(src_tokens, src_lengths = src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out = encoder_out, **kwargs)

        teacher_encoder_out, teacher_decoder_out = None, None

        if self.teacher_encoder:
            teacher_encoder_out, _ = self.teacher_encoder(src_tokens, segment_labels=None, **kwargs)
            teacher_encoder_out = teacher_encoder_out[self.args.encoder_bert_output_layer]

        if self.teacher_decoder:
            teacher_decoder_out, _ = self.teacher_decoder(prev_output_tokens, segment_labels=None, **kwargs)
            teacher_decoder_out = teacher_decoder_out[self.args.decoder_bert_output_layer]

        net_output = list()
        net_output.extend(decoder_out)
        # net_output.extend([teacher_decoder_out, encoder_out['encoder_out'], teacher_encoder_out])
        net_output.extend([teacher_decoder_out, encoder_out['encoder_out'], encoder_out['encoder_padding_mask'], teacher_encoder_out])
        return net_output

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError('--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(src_dict, args.encoder_embed_dim, args.encoder_embed_path)
            decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim, args.decoder_embed_path)

        encoder_pretrained_path = args.encoder_pretrained_checkpoints
        decoder_pretrained_path = args.decoder_pretrained_checkpoints

        if args.encoder_teacher_path:
            teacher_encoder = MaskedLMBertModel.from_pretrained(args.encoder_teacher_path)
        else:
            teacher_encoder = None

        if args.decoder_teacher_path:
            teacher_decoder = MaskedLMBertModel.from_pretrained(args.decoder_teacher_path)
        else:
            teacher_decoder = None

        encoder = BTransformerEncoder(args, src_dict, encoder_embed_tokens, pretrained_path=encoder_pretrained_path)
        decoder = BTransformerDecoder(args, tgt_dict, decoder_embed_tokens, pretrained_path=decoder_pretrained_path)

        return BTransformerModel(args, encoder, decoder, teacher_encoder, teacher_decoder)


@register_model_architecture('Btransformer', 'Btransformer')
def base_B_transformer(args):
    base_architecture(args)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)


def build_embedding(dictionary, embed_dim, path=None):
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    emb = Embedding(num_embeddings, embed_dim, padding_idx)
    # if provided, load from preloaded dictionaries
    if path:
        embed_dict = utils.parse_embedding(path)
        utils.load_embedding(embed_dict, dictionary, emb)
    return emb
