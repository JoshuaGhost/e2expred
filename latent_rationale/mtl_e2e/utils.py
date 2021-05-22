import argparse
import json
from dataclasses import dataclass
from typing import List, Any

import copy
import torch
from latent_rationale.mtl_e2e.config import E2ExPredConfig

from torch.nn.utils.rnn import pad_sequence, PackedSequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer

from latent_rationale.common.util import Example


@dataclass(eq=True, frozen=True)
class PaddedSequence:
    """A utility class for padding variable length sequences mean for RNN input
    This class is in the style of PackedSequence from the PyTorch RNN Utils,
    but is somewhat more manual in approach. It provides the ability to generate masks
    for outputs of the same input dimensions.
    The constructor should never be called directly and should only be called via
    the autopad classmethod.

    We'd love to delete this, but we pad_sequence, pack_padded_sequence, and
    pad_packed_sequence all require shuffling around tuples of information, and some
    convenience methods using these are nice to have.
    """

    data: torch.Tensor
    batch_sizes: torch.Tensor
    batch_first: bool = False

    @classmethod
    def autopad(cls, data, batch_first: bool = False, padding_value=0, device=None) -> 'PaddedSequence':
        # handle tensors of size 0 (single item)
        data_ = []
        for d in data:
            if len(d.size()) == 0:
                d = d.unsqueeze(0)
            data_.append(d)
        padded = pad_sequence(data_, batch_first=batch_first, padding_value=padding_value)
        if batch_first:
            batch_lengths = torch.LongTensor([len(x) for x in data_])
            if any([x == 0 for x in batch_lengths]):
                raise ValueError(
                    "Found a 0 length batch element, this can't possibly be right: {}".format(batch_lengths))
        else:
            # TODO actually test this codepath
            batch_lengths = torch.LongTensor([len(x) for x in data])
        return PaddedSequence(padded, batch_lengths, batch_first).to(device=device)

    # @classmethod
    # def autopad(cls, data, len_queries, max_length, batch_first, device):

    def pack_other(self, data: torch.Tensor):
        return pack_padded_sequence(data, self.batch_sizes, batch_first=self.batch_first, enforce_sorted=False)

    @classmethod
    def from_packed_sequence(cls, ps: PackedSequence, batch_first: bool, padding_value=0) -> 'PaddedSequence':
        padded, batch_sizes = pad_packed_sequence(ps, batch_first, padding_value)
        return PaddedSequence(padded, batch_sizes, batch_first)

    def cuda(self) -> 'PaddedSequence':
        return PaddedSequence(self.data.cuda(), self.batch_sizes.cuda(), batch_first=self.batch_first)

    def to(self, dtype=None, device=None, copy=False, non_blocking=False) -> 'PaddedSequence':
        # TODO make to() support all of the torch.Tensor to() variants
        return PaddedSequence(
            self.data.to(dtype=dtype, device=device, copy=copy, non_blocking=non_blocking),
            self.batch_sizes.to(device=device, copy=copy, non_blocking=non_blocking),
            batch_first=self.batch_first)

    def mask(self, mask_starts=None, on=int(0), off=int(0), device='cpu', size=None, dtype=None) -> torch.Tensor:
        if size is None:
            size = self.data.size()
        out_tensor = torch.zeros(*size, dtype=dtype)
        # TODO this can be done more efficiently
        out_tensor.fill_(off)
        if mask_starts is None:
            mask_starts = [0] * len(self.batch_sizes)
        # note to self: these are probably less efficient than explicilty populating the off values instead of the on values.
        if self.batch_first:
            for i, (mask_st, bl) in enumerate(zip(mask_starts, self.batch_sizes)):
                out_tensor[i, mask_st:bl] = on
        else:
            for i, (mask_st, bl) in enumerate(zip(mask_starts, self.batch_sizes)):
                out_tensor[mask_st:bl, i] = on
        return out_tensor.to(device)

    def unpad(self, other: torch.Tensor) -> List[torch.Tensor]:
        out = []
        for o, bl in zip(other, self.batch_sizes):
            out.append(torch.cat((o[:bl], torch.zeros(max(0, bl - len(o))))))
        return out

    def flip(self) -> 'PaddedSequence':
        return PaddedSequence(self.data.transpose(0, 1), not self.batch_first, self.padding_value)


def get_args():
    parser = argparse.ArgumentParser(description='End to End ExPred')
    parser.add_argument('--conf_fname', type=str)

    parser.add_argument('--dataset_name', type=str, choices=['movies', 'fever', 'multirc', 'short_movies'])
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--resume_snapshot', action='store_true')
    parser.add_argument('--warm_start_mtl', type=str)
    parser.add_argument('--warm_start_cls', type=str)
    parser.add_argument('--share_encoder', type=bool)

    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--eval_every', type=int, default=-1)
    parser.add_argument('--save_every', type=int, default=-1)

    parser.add_argument('--w_aux', type=float)
    parser.add_argument('--w_exp', type=float)
    parser.add_argument('--selection', type=float)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--train_on_part', type=float, default='-1')
    parser.add_argument('--decode_split', type=str, default='test')
    args = parser.parse_args()
    args = vars(args)
    conf_fname = args['conf_fname']
    with open(conf_fname, 'r') as fin:
        training_conf = json.load(fin)

    for k, v in args.items():  # values in args overwrites the ones on conf file
        if v is None:
            continue
        if k in ('selection', 'w_aux', 'w_exp'):
            training_conf['weights'][k] = v
        else:
            training_conf[k] = v
    training_conf["eval_batch_size"] = max(training_conf['batch_size'], training_conf['eval_batch_size'])

    training_conf['model_common']['num_labels'] = len(training_conf['classes'])

    mtl_conf = copy.deepcopy(training_conf['model_common'])
    if 'mtl' in training_conf:
        mtl_conf.update(training_conf['mtl'])
    training_conf['mtl'] = mtl_conf

    cls_conf = copy.deepcopy(training_conf['model_common'])
    if 'cls' in training_conf:
        cls_conf.update(training_conf['cls'])
    training_conf['cls'] = cls_conf

    model_conf = E2ExPredConfig(training_conf)
    return training_conf, model_conf


def bert_input_preprocess(examples: List[Example],
                          tokenizer: BertTokenizer,
                          max_length: int = 512,
                          device: str = None):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id
    cls_token = torch.tensor([cls_token_id])  # .to(device=device)
    sep_token = torch.tensor([sep_token_id])  # .to(device=device)
    inputs = []
    exps = []
    labels = []
    position_ids = []
    for inst in examples:
        q = inst.query
        d = inst.tokens
        exp = inst.token_labels
        labels.append(inst.label)
        if len(q) + len(d) + 2 > max_length:
            d = d[:(max_length - len(q) - 2)]
            exp = exp[:(max_length - len(q) - 2)]
        inputs.append(torch.cat([cls_token, q, sep_token, d]))
        exps.append(torch.cat([torch.Tensor([0] * (len(q) + 2)), exp]))
        position_ids.append(torch.tensor(list(range(0, len(q) + 1)) + list(range(0, len(d) + 1))))  # tokens
        # positions are counted from 1, the two 0s are for [cls] and [sep], [pad]s are also nominated as pos 0

    inputs = PaddedSequence.autopad(inputs, batch_first=True, padding_value=pad_token_id, device=device)
    positions = PaddedSequence.autopad(position_ids, batch_first=True, padding_value=0, device=device)
    exps = PaddedSequence.autopad(exps, batch_first=True, padding_value=0, device=device)
    attention_masks = inputs.mask(on=1, off=0).type(torch.float).to(device=device)
    padding_masks = inputs.mask(on=1, off=0).type(torch.bool).to(device=device)
    labels = torch.LongTensor(labels).to(device=device)
    return inputs, exps, labels, positions, attention_masks, padding_masks


# def numerify_labels(dataset, labels_mapping):
#     for exp_id, exp in enumerate(dataset):
#         dataset[exp_id] = Example(tokens=exp.tokens,
#                                   label=labels_mapping[exp.label],
#                                   token_labels=exp.token_labels,
#                                   query=exp.query,
#                                   ann_id=exp.ann_id,
#                                   docid=exp.docid)
#     return dataset


def numerify_label(ann, labels_mapping):
    return Example(tokens=ann.tokens,
                   label=labels_mapping[ann.label],
                   token_labels=ann.token_labels,
                   query=ann.query,
                   ann_id=ann.ann_id,
                   docid=ann.docid)
    return dataset


def tokenize_query_doc(example: Example, tokenizer: Any):
    if isinstance(tokenizer, BertTokenizer):
        query_tokens = tokenizer.encode(example.query, add_special_tokens=False)
    else:
        query_tokens = tokenizer.tokenize(example.query)
    tokens = []
    token_labels = []
    for token, token_label in zip(example.tokens, example.token_labels):
        if isinstance(tokenizer, BertTokenizer):
            token_pieces = tokenizer.encode(token, add_special_tokens=False)
        else:
            token_pieces = tokenizer.tokenize(token)
        tokens.extend(token_pieces)
        token_labels.extend([token_label] * len(token_pieces))
    if isinstance(tokenizer, BertTokenizer):
        return Example(query=torch.LongTensor(query_tokens),  # .type(torch.long),
                       tokens=torch.LongTensor(tokens),  # .type(torch.long),
                       token_labels=torch.Tensor(token_labels),
                       label=torch.Tensor([example.label]),
                       ann_id=example.ann_id,
                       docid=example.docid)
    else:
        return Example(query=query_tokens,
                       tokens=tokens,
                       token_labels=token_labels,
                       label=example.label,
                       ann_id=example.ann_id,
                       docid=example.docid)


def numerify_query_doc(example: Example, tokenizer: Any):
    query_ids = tokenizer.encode(example.query, add_special_tokens=False)
    token_ids = tokenizer.encode(example.tokens, add_special_tokens=False)
    return Example(query=query_ids,
                   tokens=token_ids,
                   token_labels=example.token_labels,
                   label=example.label,
                   ann_id=example.ann_id,
                   docid=example.docid)
