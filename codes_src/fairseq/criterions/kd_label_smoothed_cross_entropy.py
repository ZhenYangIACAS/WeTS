# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion

from torch.nn import MSELoss


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('kd_label_smoothed_cross_entropy')
class KdLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    """
    to distill the knowledge from the Bert model into NMT
    papers: "Acquiring Knowledge from pre-trained Model to Neural Machine Translation"

    """

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.kd_encoder_alpha = args.kd_encoder_alpha
        self.kd_decoder_alpha = args.kd_decoder_alpha

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--kd-encoder-alpha', default=0.5, type=float, metavar='D',
                            help='alpha for encoder bert knowledge distillation, 0 means no bert kd')
        parser.add_argument('--kd-decoder-alpha', default=0.5, type=float, metavar='D',
                            help='alpha for decoder bert knowledge distillation, 0 means no bert kd')
        # fmt: on

    def forward_v2(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # net_output = model(**sample['net_input'])
        net_output = model(**sample['net_input'])
        decoder_out = net_output[:2]
        teacher_decoder_out = net_output[2]
        encoder_out = net_output[3]
        teacher_encoder_out = net_output[4]

        loss, nll_loss = self.compute_loss_v2(model,
                                           decoder_out,
                                           teacher_decoder_out,
                                           encoder_out,
                                           teacher_encoder_out,
                                           sample, reduce=reduce)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # net_output = model(**sample['net_input'])
        net_output = model(**sample['net_input'])
        decoder_out = net_output[:2]
        teacher_decoder_out = net_output[2]
        encoder_out = net_output[3]
        encoder_padding_mask = net_output[4]
        teacher_encoder_out = net_output[5]

        loss, nll_loss = self.compute_loss_v3(model,
                                           decoder_out,
                                           teacher_decoder_out,
                                           encoder_out,
                                           encoder_padding_mask,
                                           teacher_encoder_out,
                                           sample,
                                           reduce=reduce)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model,
                     decoder_out,
                     teacher_decoder_out,
                     encoder_out,
                     teacher_encoder_out,
                     sample,
                     reduce=True):

        lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, decoder_out).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        decoder_kd_error, encoder_kd_error = 0.0, 0.0
        if teacher_decoder_out is not None:
            decoder_out_view = decoder_out[0].view(-1, decoder_out.size(-1))
            teacher_decoder_out_view = teacher_decoder_out.view(-1, teacher_decoder_out.size(-1))
            # decoder_kd_loss = MSELoss(reduce=True, size_average=True)
            decoder_kd_loss = MSELoss(reduction='mean')
            decoder_kd_error = decoder_kd_loss(decoder_out_view, teacher_decoder_out_view)
        if teacher_encoder_out is not None:
            encoder_out_view = encoder_out.view(-1, encoder_out.size(-1))
            teacher_encoder_out_view = teacher_encoder_out.view(-1, teacher_encoder_out.size(-1))
            # encoder_kd_loss = MSELoss(reduce=True, size_average=True)
            encoder_kd_loss = MSELoss(reduction='mean')
            encoder_kd_error = encoder_kd_loss(encoder_out_view, teacher_encoder_out_view)
        
        # print('encoder_kd_error', encoder_kd_error)
        # print('decoder_kd_error', decoder_kd_error)
        #loss = loss + self.kd_encoder_alpha * encoder_kd_error + self.kd_decoder_alpha * decoder_kd_error
        loss = (1 - self.kd_encoder_alpha) * loss + self.kd_encoder_alpha * encoder_kd_error + self.kd_decoder_alpha * decoder_kd_error

        return loss, nll_loss

    def compute_loss_v2(self, model,
                     decoder_out,
                     teacher_decoder_out,
                     encoder_out,
                     teacher_encoder_out,
                     sample,
                     reduce=True):
        # to average at the sentence level, and sum at the batch level
        lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, decoder_out).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        decoder_kd_error, encoder_kd_error = 0.0, 0.0
        if teacher_decoder_out is not None:
            decoder_out_view = decoder_out[0].view(-1, decoder_out.size(-1))
            teacher_decoder_out_view = teacher_decoder_out.view(-1, teacher_decoder_out.size(-1))
            decoder_kd_loss = MSELoss(reduction='sum')
            decoder_kd_error = decoder_kd_loss(decoder_out_view, teacher_decoder_out_view)
            decoder_kd_error = decoder_kd_error / decoder_out[0].size(1)
        if teacher_encoder_out is not None:
            encoder_out_view = encoder_out.view(-1, encoder_out.size(-1))
            teacher_encoder_out_view = teacher_encoder_out.view(-1, teacher_encoder_out.size(-1))
            encoder_kd_loss = MSELoss(reduction='sum')
            encoder_kd_error = encoder_kd_loss(encoder_out_view, teacher_encoder_out_view)
            encoder_kd_error = encoder_kd_error / encoder_out.size(1)
        
        # print('encoder_kd_error', encoder_kd_error)
        # print('decoder_kd_error', decoder_kd_error)
        loss = loss + self.kd_encoder_alpha * encoder_kd_error + self.kd_decoder_alpha * decoder_kd_error

        return loss, nll_loss

    def compute_loss_v3(self, model,
                     decoder_out,
                     teacher_decoder_out,
                     encoder_out,
                     encoder_padding_mask,
                     teacher_encoder_out,
                     sample,
                     reduce=True):
        # to average at the sentence level, and sum at the batch level
        lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, decoder_out).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        decoder_kd_error, encoder_kd_error = 0.0, 0.0
        if teacher_decoder_out is not None:
            decoder_out_view = decoder_out[1]['inner_states'][-1]
            decoder_out_view = decoder_out_view.view(-1, decoder_out_view.size(-1))
            teacher_decoder_out_view = teacher_decoder_out.view(-1, teacher_decoder_out.size(-1))
            decoder_kd_loss = MSELoss(reduction='sum')
            decoder_kd_error = decoder_kd_loss(decoder_out_view, teacher_decoder_out_view)
            # decoder_kd_error = decoder_kd_error / decoder_out[0].size(1)
            decoder_kd_error = decoder_kd_error / decoder_out_view.size(0)

        if teacher_encoder_out is not None:
            # T * B * C -> B * T * C
            encoder_out = encoder_out.transpose(0,1).contiguous()
            teacher_encoder_out = teacher_encoder_out.transpose(0,1).contiguous()
            if encoder_padding_mask is not None:
                encoder_out *= 1 - encoder_padding_mask.unsqueeze(-1).type_as(encoder_out)
                teacher_encoder_out *= 1 - encoder_padding_mask.unsqueeze(-1).type_as(teacher_encoder_out)
            encoder_out_view = encoder_out.view(-1, encoder_out.size(-1))
            teacher_encoder_out_view = teacher_encoder_out.view(-1, teacher_encoder_out.size(-1))
            encoder_kd_loss = MSELoss(reduction='sum')
            encoder_kd_error = encoder_kd_loss(encoder_out_view, teacher_encoder_out_view)
            # encoder_kd_error = encoder_kd_error / encoder_out.size(1)
            encoder_kd_error = encoder_kd_error / encoder_out_view.size(0)
        
        # print('encoder_kd_error', encoder_kd_error)
        # print('decoder_kd_error', decoder_kd_error)
        loss = loss + self.kd_encoder_alpha * encoder_kd_error + self.kd_decoder_alpha * decoder_kd_error
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
