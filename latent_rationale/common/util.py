import json
from collections import namedtuple

import numpy as np
import random
import torch
from torch import nn
from torch.nn.init import _calculate_fan_in_and_fan_out

from latent_rationale.nn.bow_encoder import BOWEncoder
from latent_rationale.nn.cnn_encoder import CNNEncoder
from latent_rationale.nn.lstm_encoder import LSTMEncoder
from latent_rationale.nn.rcnn_encoder import RCNNEncoder

Example = namedtuple("Example", ["tokens", "label", "token_labels", 'query', 'docid', 'ann_id'])


def make_kv_string(d: object) -> object:
    out = []
    for k, v in d.items():
        if isinstance(v, float):
            out.append("{} {:.4f}".format(k, v))
        else:
            out.append("{} {}".format(k, v))

    return " ".join(out)


def get_encoder(layer, in_features, hidden_size, bidirectional=True):
    """Returns the requested layer."""
    if layer == "lstm":
        return LSTMEncoder(in_features, hidden_size,
                           bidirectional=bidirectional)
    elif layer == "rcnn":
        return RCNNEncoder(in_features, hidden_size,
                           bidirectional=bidirectional)
    elif layer == "bow":
        return BOWEncoder()
    elif layer == "cnn":
        return CNNEncoder(
            embedding_size=in_features, hidden_size=hidden_size,
            kernel_size=5)
    else:
        raise ValueError("Unknown layer")


def get_z_stats(z=None, mask=None):
    """
    Computes statistics about how many zs are
    exactly 0, continuous (between 0 and 1), or exactly 1.

    :param z:
    :param mask: mask in [B, T]
    :return:
    """

    z = torch.where(mask, z, z.new_full([1], 1e2))

    num_0 = (z == 0.).sum().item()
    num_c = ((z > 0.) & (z < 1.)).sum().item()
    num_1 = (z == 1.).sum().item()

    total = num_0 + num_c + num_1
    mask_total = mask.sum().item()

    assert total == mask_total, "total mismatch"
    return num_0, num_c, num_1, mask_total


def xavier_uniform_n_(w, gain=1., n=4):
    """
    Xavier initializer for parameters that combine multiple matrices in one
    parameter for efficiency. This is e.g. used for GRU and LSTM parameters,
    where e.g. all gates are computed at the same time by 1 big matrix.
    :param w:
    :param gain:
    :param n:
    :return:
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out = fan_out // n
        std = gain * np.math.sqrt(2.0 / (fan_in + fan_out))
        a = np.math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)


def initialize_model_(model):
    """
    Model initialization.

    :param model:
    :return:
    """
    # Custom initialization
    print("Glorot init")
    for name, p in model.named_parameters():
        if "bert_model" in name or name.startswith("embed") or "lagrange" in name:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))
        elif "lstm" in name and len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier_n", name, p.shape))
            xavier_uniform_n_(p)
        elif len(p.shape) > 1:
            print("{:10s} {:20s} {}".format("xavier", name, p.shape))
            torch.nn.init.xavier_uniform_(p)
        elif "bias" in name:
            print("{:10s} {:20s} {}".format("zeros", name, p.shape))
            torch.nn.init.constant_(p, 0.)
        else:
            print("{:10s} {:20s} {}".format("unchanged", name, p.shape))


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def print_parameters(model):
    """Prints model parameters"""
    total = 0
    for name, p in model.named_parameters():
        total += np.prod(p.shape)
        print("{:24s} {:12s} requires_grad={}".format(name, str(list(p.shape)),
                                                      p.requires_grad))
    print("\nTotal parameters: {}\n".format(total))


def get_minibatch(data, batch_size, shuffle=False, train_on_part=-1):
    """Return minibatches, optional shuffling"""
    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch

    batch = []

    # yield minibatches
    if train_on_part == -1:
        data_to_return = data
    else:
        data_to_return = data[:int(len(data) * train_on_part)]

    for example in data_to_return:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch
            batch = []

    # in case there is something left
    if len(batch) > 0:
        yield batch


# def get_minibatch(data, batch_size, shuffle=False):
#     """Return minibatches, optional shuffling"""
#     if shuffle:
#         print("Shuffling training data")
#         random.shuffle(data)  # shuffle training data each epoch
#
#     batch = []
#
#     # yield minibatches
#     for example in data:
#         batch.append(example)
#
#         if len(batch) == batch_size:
#             yield batch
#             batch = []
#
#     # in case there is something left
#     if len(batch) > 0:
#         yield batch


def write_jsonl(jsonl, output_file):
    with open(output_file, 'w') as of:
        for js in jsonl:
            as_str = json.dumps(js, sort_keys=True)
            of.write(as_str)
            of.write('\n')
