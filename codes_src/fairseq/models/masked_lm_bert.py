# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    SinusoidalPositionalEmbedding,
    TransformerSentenceEncoder,
)

from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.data.legacy.masked_lm_dictionary import MaskedLMDictionary

class BertConfig(object):
    def __init__(self):
        self.encoder_embed_dim=512
        self.share_encoder_input_output_embed=True
        self.no_token_positional_embeddings=False
        self.encoder_learned_pos = False
        self.num_segment=0
        self.encoder_layers = 12
        self.encoder_attention_heads = 8
        self.encoder_ffn_embed_dim = 2048
        self.bias_kv = False
        self.zero_attn = False
        self.sentence_class_num = 2
        self.sent_loss = False
        self.apply_bert_init = True
        self.activation_fn = 'gelu'
        self.pooler_activation_fn = 'tanh'
        self.encoder_normalize_before = False
        self.dropout = 0.1
        self.attention_dropout = 0.1
        self.act_dropout = 0.0
        self.tokens_per_sample = 1024
        #self.dict_path='/dockerdata/zieenyang/corpus/data_fairseq/zhyue-new-standard-sp-pretrain-fit-xlm/processed_only_para/dict.yue.txt'
        self.dict_path='/apdcephfs/share_1157259/users/zieenyang/corpus/data_xlm_fairseq/zh-yue-new-standard-sp10k-cs-zh-zh-200w/dict.yue.txt'

class MaskedLMBertModel(BaseFairseqModel):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """
    def __init__(self, args, encoder):
        super().__init__()
        self.args = args
        self.encoder = encoder

        # if specified then apply bert initialization on the model. We need
        # to explictly call this to make sure that the output embeddings
        # and projection layers are also correctly initialized
        if getattr(args, 'apply_bert_init', False):
            self.apply(init_bert_params)

    def forward(self, src_tokens, segment_labels=None, **kwargs):
        return self.encoder(src_tokens, segment_labels, **kwargs)

    def max_positions(self):
        return self.encoder.max_positions

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        print("Model args: ", args)

        encoder = MaskedLMEncoder(args, task.dictionary)
        return cls(args, encoder)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        config = BertConfig()
        dictionary = MaskedLMDictionary.load(config.dict_path)
        if not hasattr(config, 'max_positions'):
            config.max_positions = config.tokens_per_sample

        print('Model config for pretrained Bert: ', config.__dict__)
        encoder = MaskedLMEncoder(config, dictionary)
        model = cls(config, encoder)
        state_dict = torch.load(pretrained_model_name_or_path, map_location='cpu')
        state_dict = state_dict['model']
        # print('load state_dict: ', state_dict.keys())
        model_state_dict = model.state_dict()
        # print('model state dict: ', model_state_dict.keys())
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata
        
        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    # print('load modules', prefix + name)
                    load(child, prefix + name + '.')

        start_prefix = ''
        load(model, prefix=start_prefix)
        print('load the pretrained bert done!')

        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: ", 
                model.__class__.__name__, missing_keys)
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in: ", model.__class__.__name__, unexpected_keys)
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        return model

class MaskedLMEncoder(FairseqEncoder):
    """
    Encoder for Masked Language Modelling.
    """

    def __init__(self, args, dictionary):
        super().__init__(dictionary)

        self.padding_idx = dictionary.pad()
        self.vocab_size = dictionary.__len__()
        self.max_positions = args.max_positions

        self.sentence_encoder = TransformerSentenceEncoder(
            padding_idx=self.padding_idx,
            vocab_size=self.vocab_size,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=self.max_positions,
            num_segments=args.num_segment,
            use_position_embeddings=not args.no_token_positional_embeddings,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_bert_init=args.apply_bert_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.encoder_learned_pos,
            add_bias_kv=args.bias_kv,
            add_zero_attn=args.zero_attn,
        )

        self.share_input_output_embed = args.share_encoder_input_output_embed
        self.embed_out = None
        self.sentence_projection_layer = None
        self.sentence_out_dim = args.sentence_class_num
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = not getattr(args, 'remove_head', False)

        self.masked_lm_pooler = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.pooler_activation = utils.get_activation_fn(args.pooler_activation_fn)

        self.lm_head_transform_weight = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(self.vocab_size))

            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    args.encoder_embed_dim,
                    self.vocab_size,
                    bias=False
                )

            if args.sent_loss:
                self.sentence_projection_layer = nn.Linear(
                    args.encoder_embed_dim,
                    self.sentence_out_dim,
                    bias=False
                )

    def forward(self, src_tokens, segment_labels=None, **unused):
        """
        Forward pass for Masked LM encoder. This first computes the token
        embedding using the token embedding matrix, position embeddings (if
        specified) and segment embeddings (if specified).

        Here we assume that the sentence representation corresponds to the
        output of the classification_token (see bert_task or cross_lingual_lm
        task for more details).
        Args:
            - src_tokens: B x T matrix representing sentences
            - segment_labels: B x T matrix representing segment label for tokens
        Returns:
            - a tuple of the following:
                - logits for predictions in format B x T x C to be used in
                  softmax afterwards
                - a dictionary of additional data, where 'pooled_output' contains
                  the representation for classification_token and 'inner_states'
                  is a list of internal model states used to compute the
                  predictions (similar in ELMO). 'sentence_logits'
                  is the prediction logit for NSP task and is only computed if
                  this is specified in the input arguments.
        """

        inner_states, sentence_rep = self.sentence_encoder(src_tokens, segment_labels)

        x = inner_states[-1].transpose(0, 1)
        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        pooled_output = self.pooler_activation(self.masked_lm_pooler(sentence_rep))

        # project back to size of vocabulary
        if self.share_input_output_embed \
                and hasattr(self.sentence_encoder.embed_tokens, 'weight'):
            x = F.linear(x, self.sentence_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)

        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias
        sentence_logits = None
        if self.sentence_projection_layer:
            sentence_logits = self.sentence_projection_layer(pooled_output)

        return inner_states, pooled_output
        #return x, {
        #    'inner_states': inner_states,
        #    'pooled_output': pooled_output,
        #    'sentence_logits': sentence_logits
        #}

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        if isinstance(
                self.sentence_encoder.embed_positions,
                SinusoidalPositionalEmbedding
        ):
            state_dict[
                name + '.sentence_encoder.embed_positions._float_tensor'
            ] = torch.FloatTensor(1)
        if not self.load_softmax:
            for k in list(state_dict.keys()):
                if (
                    "embed_out.weight" in k or
                    "sentence_projection_layer.weight" in k or
                    "lm_output_learned_bias" in k
                ):
                    del state_dict[k]
        return state_dict
