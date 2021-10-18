# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion

import torch.nn.functional as F
import torch

def compute_cross_entropy_loss(logits, targets, ignore_index=-100):
    """
    Function to compute the cross entropy loss. The default value of
    ignore_index is the same as the default value for F.cross_entropy in
    pytorch.
    """
    assert logits.size(0) == targets.size(-1), \
        "Logits and Targets tensor shapes don't match up"

    loss = F.nll_loss(
        F.log_softmax(logits, -1, dtype=torch.float32),
        targets,
        reduction="sum",
        ignore_index=ignore_index,
    )
    return loss

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


@register_criterion('mlm_lsce_multi_task_loss')
class MlmLabelSmoothedCrossEntropyMtaskCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True, use_mlm_loss=False):
        if use_mlm_loss:
            loss, sample_size, logging_output = self._forward_mlm_loss(model, sample, reduce)
        else:
            loss, sample_size, logging_output = self._forward_lsce_loss(model, sample, reduce)

        return loss, sample_size, logging_output
    

    def _forward_mlm_loss(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        model_output = model.encoder(**sample["net_input"])
        lm_logits = model_output['lm_out']

        # reshape lm_logits from (N,T,C) to (N*T,C)
        lm_logits = lm_logits.view(-1, lm_logits.size(-1))
        lm_targets = sample['lm_target'].view(-1)
        lm_loss = compute_cross_entropy_loss(
            lm_logits, lm_targets, self.padding_idx)

        # compute the number of tokens for which loss is computed. This is used
        # to normalize the loss
        ntokens = utils.strip_pad(lm_targets, self.padding_idx).numel()
        loss = lm_loss / ntokens
        nsentences = sample['nsentences']

        sample_size = 1
        sentence_loss=None
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'lm_loss': utils.item(lm_loss.data) if reduce else lm_loss.data,
            # sentence loss is not always computed
            'sentence_loss': (
                (
                    utils.item(sentence_loss.data) if reduce
                    else sentence_loss.data
                ) if sentence_loss is not None else 0.0
            ),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def _forward_lsce_loss(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
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
