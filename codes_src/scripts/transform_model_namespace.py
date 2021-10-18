#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import collections
import torch
import os
import re
from torch.serialization import default_restore_location


def change_checkpoints(inputs):
    """Loads checkpoints from inputs and returns a model with averaged weights.

    Args:
      inputs: An iterable of string paths of checkpoints to load from.

    Returns:
      A dict of string keys mapping to various values. The 'model' key
      from the returned dict should correspond to an OrderedDict mapping
      string parameter names to torch Tensors.
    """
    params_dict = collections.OrderedDict()
    params_keys = None
    new_state = None
    num_models = len(inputs)
    assert num_models == 1, num_models

    for f in inputs:
        state = torch.load(
            f,
            map_location=(
                lambda s, l: default_restore_location(s, 'cpu')
            ),
        )
        # Copies over the settings from the first checkpoint
        if new_state is None:
            new_state = state

        model_params = state['model']

        model_params_keys = list(model_params.keys())
        if params_keys is None:
            params_keys = model_params_keys
        elif params_keys != model_params_keys:
            raise KeyError(
                'For checkpoint {}, expected list of params: {}, '
                'but found: {}'.format(f, params_keys, model_params_keys)
            )

        for k in params_keys:
            print(k)
            new_k = k
            if 'encoders' in k or 'decoders' in k:
                new_k = re.sub(r's.[a-z]{1,}.', '.', k, count=1)
                print(new_k)
            p = model_params[k]
            if isinstance(p, torch.HalfTensor):
                p = p.float()
            if k not in params_dict:
                params_dict[new_k] = p.clone()
                # NOTE: clone() is needed in case of p is a shared parameter

    new_state['model'] = params_dict
    return new_state


def main():
    parser = argparse.ArgumentParser(
        description='Tool to average the params of input checkpoints to '
                    'produce a new checkpoint',
    )
    # fmt: off
    parser.add_argument('--inputs', required=True, nargs='+',
                        help='Input checkpoint file paths.')
    parser.add_argument('--output', required=True, metavar='FILE',
                        help='Write the new checkpoint containing the averaged weights to this path.')

    args = parser.parse_args()
    new_state = change_checkpoints(args.inputs)
    torch.save(new_state, args.output)
    print('Finished writing averaged checkpoint to {}.'.format(args.output))


if __name__ == '__main__':
    main()
