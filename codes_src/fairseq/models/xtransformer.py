# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#

from collections import OrderedDict

from fairseq import utils
from fairseq.models import FairseqMultiModel, register_model, register_model_architecture, BaseFairseqModel

from fairseq.models.transformer import (
    base_architecture,
    Embedding,
    TransformerEncoder,
    TransformerDecoder,
    TransformerModel,
)

import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict
import os
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

    state = checkpoint_utils.load_checkpoint_to_cpu(pretrained_xlm_checkpoint)
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

class XTransformerEncoder(TransformerEncoder):

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


class XTransformerDecoder(TransformerDecoder):

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


@register_model('xtransformer')
class XTransformerModel(BaseFairseqModel):
    def __init__(self, encoders, decoders, eval_lang_pair=None):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)
        self.tgt_key = None
        if eval_lang_pair is not None:
            self.source_lang = eval_lang_pair.split('-')[0]
            self.target_lang = eval_lang_pair.split('-')[1]

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif hasattr(self, 'decoders'):
            return self.decoders[self.tgt_key].get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def max_positions(self):
        return None

    def max_decoder_positions(self):
        return min(decoder.max_positions() for decoder in self.decoders.values())

    def forward(self, src_tokens, src_lengths, prev_output_tokens, src_key, tgt_key, positions=None):
        encoder_out = self.encoders[src_key](src_tokens, src_lengths)
        decoder_out = self.decoders[tgt_key](prev_output_tokens, encoder_out, positions=positions)
        self.tgt_key = tgt_key
        return decoder_out

    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--share-encoders', action='store_true',
                            help='share encoders across languages')
        parser.add_argument('--share-decoders', action='store_true',
                            help='share decoders across languages')
        parser.add_argument('--encoder-pretrained-checkpoints', type=str, default=None, metavar='STR', help='path for reload the pretrained encoder checkpoints, with the format: lang1:path1,lang2:path2')
        parser.add_argument('--decoder-pretrained-checkpoints', type=str, default=None, metavar='STR', help='path for reload the pretrained decoder checkpoints, with the format: lang1:path1,lang2:path2')

    @classmethod
    def build_model(cls, args, task):
        langs = [lang for lang in args.langs]

        embed_tokens = {}
        for lang in langs:
            if len(embed_tokens) == 0 or args.share_all_embeddings is False:
                embed_token = build_embedding(
                    task.dicts[lang], args.encoder_embed_dim, args.encoder_embed_path        
                )
                embed_tokens[lang] = embed_token
            else:
                embed_tokens[lang] = embed_tokens[langs[0]]

        args.share_decoder_input_output_embed = True
        # added by zieenyang  at 20191009
        encoder_pretrained_path, decoder_pretrained_path = {}, {}

        if args.encoder_pretrained_checkpoints:
            path_list = args.encoder_pretrained_checkpoints.split(',')
            for lang_path in path_list:
                lang_path_list = lang_path.strip().split(':')
                assert len(lang_path_list) == 2, lang_path_list
                lang, path = lang_path_list[0].strip(),lang_path_list[1].strip()
                assert lang in args.source_langs, 'language {} not in source_langs'.format(lang)
                assert lang not in encoder_pretrained_path, 'language {} has two pretrained checkpoints'.format(lang)
                encoder_pretrained_path[lang] = path

        if args.decoder_pretrained_checkpoints:
            path_list = args.decoder_pretrained_checkpoints.split(',')
            for lang_path in path_list:
                lang_path_list = lang_path.strip().split(':')
                assert len(lang_path_list) == 2, lang_path_list
                lang, path = lang_path_list[0].strip(),lang_path_list[1].strip()
                assert lang in args.target_langs, 'language {} not in source_langs'.format(lang)
                assert lang not in decoder_pretrained_path, 'language {} has two pretrained checkpoints'.format(lang)
                decoder_pretrained_path[lang] = path
        
        encoders, decoders = {}, {}

        for lang in langs:
            encoder_embed_tokens = embed_tokens[lang]
            decoder_embed_tokens = encoder_embed_tokens
            if lang in args.source_langs:
                if len(encoders) == 0 or args.share_encoders is False:
                    if lang in encoder_pretrained_path:
                        encoder = XTransformerEncoder(args, task.dicts[lang], encoder_embed_tokens, pretrained_path=encoder_pretrained_path[lang])
                    else:
                        encoder = XTransformerEncoder(args, task.dicts[lang], encoder_embed_tokens)
                    encoders[lang] = encoder
                else:
                    encoders[lang] =list(encoders.values())[0]

            if lang in args.target_langs:
                if len(decoders) == 0 or args.share_decoders is False:
                    if lang in decoder_pretrained_path:
                        decoder = XTransformerDecoder(args, task.dicts[lang], decoder_embed_tokens, pretrained_path=decoder_pretrained_path[lang])
                    else:
                        decoder = XTransformerDecoder(args, task.dicts[lang], decoder_embed_tokens)
                    decoders[lang] = decoder 
                else:
                    decoders[lang] = list(decoders.values())[0]
        return XTransformerModel(encoders, decoders, args.eval_lang_pair)
    
    @property
    def decoder(self):
        return self.decoders[self.target_lang]

    @property
    def encoder(self):
        return self.encoders[self.source_lang]


@register_model_architecture('xtransformer', 'xtransformer')
def base_x_transformer(args):
    base_architecture(args)
    args.share_encoders = getattr(args, 'share_encoders', False)
    args.share_decoders = getattr(args, 'share_decoders', False)
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
