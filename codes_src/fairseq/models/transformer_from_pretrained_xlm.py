# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Dict

from fairseq import checkpoint_utils
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary
from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import (
    TransformerDecoder,
    TransformerEncoder,
    TransformerModel,
    base_architecture as transformer_base_architecture,
)
from fairseq.models.transformer import DEFAULT_MAX_SOURCE_POSITIONS, DEFAULT_MAX_TARGET_POSITIONS, Embedding

from collections import OrderedDict

##################################################
# transformer model from pretrained xlm model
# Author: zieenyang
# Data:20190912
##################################################

@register_model("transformer_from_pretrained_xlm")
class TransformerFromPretrainedXLMModel(TransformerModel):

    def __init__(self, encoder, decoder, model_lang_pairs):
        super().__init__(encoder, decoder)
        self.model_lang_pairs = model_lang_pairs

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        TransformerModel.add_args(parser)
        parser.add_argument(
            "--pretrained-xlm-checkpoint",
            type=str,
            metavar="STR",
            help="XLM model to use for initializing transformer encoder and/or decoder",
        )
        parser.add_argument(
            "--init-encoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into decoder",
        )
        parser.add_argument(
            "--init-decoder-only",
            action="store_true",
            help="if set, don't load the XLM weights and embeddings into encoder",
        )
        parser.add_argument(
            "--src-segment-nums",
            type=int,
            default=0,
            help='segment numbers for encoders'
        )

        parser.add_argument(
            "--tgt-segment-nums",
            type=int,
            default=0,
            help='segment numbers for decoders'
        )
    @classmethod
    def build_model(cls, args, task, cls_dictionary=MaskedLMDictionary):
        assert hasattr(args, "pretrained_xlm_checkpoint"), (
            "You must specify a path for --pretrained-xlm-checkpoint to use "
            "--arch transformer_from_pretrained_xlm"
        )
        assert isinstance(task.source_dictionary, cls_dictionary) and isinstance(
            task.target_dictionary, cls_dictionary
        ), (
            "You should use a MaskedLMDictionary when using --arch "
            "transformer_from_pretrained_xlm because the pretrained XLM model "
            "was trained using data binarized with MaskedLMDictionary. "
            "For translation, you may want to use --task "
            "translation_from_pretrained_xlm"
        )
        assert not (
            getattr(args, "init_encoder_only", False)
            and getattr(args, "init_decoder_only", False)
        ), "Only one of --init-encoder-only and --init-decoder-only can be set."

        base_architecture(args)

        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        lang_pairs = task.model_lang_pairs

        return TransformerFromPretrainedXLMModel(encoder, decoder, lang_pairs)

    def max_positions(self):
        return {
            key: (self.encoder.max_positions(), self.decoder.max_positions())
            for key in self.model_lang_pairs
        }
        # assert 1==2, 'running here, model_lang_pairs {}'.format(self.model_lang_pairs)
        # return OrderedDict([(key, (self.encoder.max_positions(), self.decoder.max_positions()))for key in self.model_lang_pairs])

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoderFromPretrainedXLM(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderFromPretrainedXLM(args, tgt_dict, embed_tokens)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, src_segment_tokens, tgt_segment_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, src_segment_tokens=src_segment_tokens, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, tgt_segment_tokens=tgt_segment_tokens, **kwargs)

        return decoder_out

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
        # added by zieenyang
        for search_key in ["embed_tokens", "embed_positions", "layers"] + ['segment_embeddings']:
            if search_key in key:
                subkey = key[key.find(search_key):]
                #assert subkey in state_dict, (
                #    "{} Transformer encoder / decoder "
                #    "state_dict does not contain {}. Cannot "
                #    "load {} from pretrained XLM checkpoint "
                #    "{} into Transformer.".format(
                #        str(state_dict.keys()),
                #        subkey, key, pretrained_xlm_checkpoint)
                #    )

                if subkey not in state_dict:
                    print('| {} subkey not found in state_dict {}'.format(subkey, str(state_dict.keys())))
                else:
                    print('| {} subkey found in state_dict'.format(subkey))
                    state_dict[subkey] = xlm_state_dict[key]
    return state_dict


class TransformerEncoderFromPretrainedXLM(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        if getattr(args, 'init_decoder_only', False):
            # Don't load XLM weights for encoder if --init-decoder-only
            return

        assert hasattr(args, "pretrained_xlm_checkpoint"), (
            "--pretrained-xlm-checkpoint must be specified to load Transformer "
            "encoder from pretrained XLM"
        )
        xlm_loaded_state_dict = upgrade_state_dict_with_xlm_weights(
            state_dict=self.state_dict(),
            pretrained_xlm_checkpoint=args.pretrained_xlm_checkpoint,
        )

        self.load_state_dict(xlm_loaded_state_dict, strict=True)


class TransformerDecoderFromPretrainedXLM(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        if getattr(args, 'init_encoder_only', False):
            # Don't load XLM weights for decoder if --init-encoder-only
            return
        assert hasattr(args, "pretrained_xlm_checkpoint"), (
            "--pretrained-xlm-checkpoint must be specified to load Transformer "
            "decoder from pretrained XLM"
        )

        xlm_loaded_state_dict = upgrade_state_dict_with_xlm_weights(
            state_dict=self.state_dict(),
            pretrained_xlm_checkpoint=args.pretrained_xlm_checkpoint,
        )
        self.load_state_dict(xlm_loaded_state_dict, strict=True)


@register_model_architecture(
    "transformer_from_pretrained_xlm", "transformer_from_pretrained_xlm"
)
def base_architecture(args):
    # added by zieenyang for languge-specific embedding
    args.source_segment_nums = getattr(args, 'src_segment_nums', 2)
    args.target_segment_nums = getattr(args, 'tgt_segment_nums', 2)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.decoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.decoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    transformer_base_architecture(args)
