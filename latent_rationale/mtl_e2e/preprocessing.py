import torch
from typing import List
from transformers import BertTokenizer
from latent_rationale.mtl_e2e.utils import PaddedSequence


def preprocess(q: List[int],
               d: List[int],
               max_length: int,
               tokenizer: BertTokenizer,
               device='cpu'):
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    cls_token = torch.tensor([cls_token_id]).to(device)
    sep_token = torch.tensor([sep_token_id]).to(device)
    input_tensors = []
    selector_mask_starts = []
    if len(q) + len(d) + 2 > max_length:
        d = d[:(max_length - len(q) - 2)]
    input_tensors.append(torch.cat([cls_token, q, sep_token, d]))
    selector_mask_starts.append(len(q) + 2)
    bert_input = PaddedSequence.autopad(input_tensors, batch_first=True, padding_value=self.pad_token_id,
                                        device=device)
    attention_mask = bert_input.mask(on=1., off=0., device=device)
    return bert_input, attention_mask


