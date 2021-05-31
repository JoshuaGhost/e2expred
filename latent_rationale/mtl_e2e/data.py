import pickle
import os
import torch
from torch.utils.data import DataLoader
import multiprocessing
from functools import partial
from tqdm import tqdm
from transformers import BertTokenizer

from latent_rationale.common.util import Example
from latent_rationale.mtl_e2e.utils import tokenize_query_doc, bert_input_preprocess, numerify_label


class MTLDataLoader(DataLoader):

    def _preprocess(self, raw_data, num_workers=None):
        print("Preprocessing the dataset for the first time.")
        num_workers = None
        if num_workers is not None:
            # with multiprocessing.Pool(processes=num_workers) as pool:
            #     data = pool.map(partial(numerify_label, labels_mapping=self.label_name_to_id), raw_data)
            # print('done with numerifying labels')
            preprocess_batch_size = 512
            data = []
            for preprocess_batch_start in range(0, len(raw_data), preprocess_batch_size):
                _raw_data = raw_data[preprocess_batch_start: preprocess_batch_start+preprocess_batch_size]
                print(len(_raw_data))
                with multiprocessing.Pool(processes=num_workers) as pool:
                    data += pool.map(partial(tokenize_query_doc,
                                             labels_mapping=self.label_name_to_id,
                                             tokenizer=self.tokenizer),
                                     _raw_data)
            print('done with tokenization')
        else:
            data = []
            for instance in tqdm(raw_data):
                data.append(tokenize_query_doc(instance, self.label_name_to_id, self.tokenizer))
            # data = [tokenize_query_doc(instance, self.label_name_to_id, self.tokenizer)
            #         for instance in raw_data]
        return data

    def __init__(self, raw_data, label_name_to_id, tokenizer, max_length,
                 batch_size, shuffle, num_workers, cache_fname=None):
        self.raw_data = raw_data
        self.cache_fname = cache_fname
        self.label_name_to_id = label_name_to_id
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.cache_fname is None or not os.path.isfile(self.cache_fname):
            self.data = self._preprocess(self.raw_data, num_workers)
            if self.cache_fname is not None:
                import sys
                if sys.getsizeof(self.data) > (100 * 1024 * 1024 * 1024):  # 100GB
                    with open(self.cache_fname, 'w') as fout:
                        fout.write('preprocessed data too large (>100GB), will not cache\n')
                else:
                    with open(self.cache_fname, 'wb+') as fout:
                        pickle.dump(self.data, fout)
                    print(f'preprocessed dataset dumped at {self.cache_fname}')
        else:
            print(f'Preprocessed dataset found at {self.cache_fname}, loading...')
            with open(self.cache_fname, 'rb') as fin:
                self.data = pickle.load(fin)
        super(MTLDataLoader, self).__init__(self.data, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers, pin_memory=True,
                                            collate_fn=self.bert_input_preprocess)

    def __getitem__(self, index):
        # input, exp, label = bert_input_preprocess(self.data[index], tokenizer=self.tokenizer)
        # input, positions, attention_masks = pad_to_max_length(input, max_length=max_length)
        # inputs, exps, labels, positions, attention_masks = \
        #     bert_input_preprocess(data, self.tokenizer, max_length, device='cpu')
        # data = list(zip(inputs.data, exps.data, labels, positions.data, attention_masks.data))
        (
            tokens,
            exp_labels, cls_label,
            position_ids, attention_mask
        ) = self.bert_input_preprocess(
            self.data[index]
        )
        print(f'input.shape={input.shape}'
              f'exp_lables.shape={exp_labels.shape}'
              f'cls_lable.shape={cls_label.shape}'
              f'position_ids.shape={position_ids.shape}'
              f'attention_mask.shape={attention_mask.shape}')
        return input, exp_labels, cls_label, position_ids, attention_mask

    def __len__(self):
        return len(self.data)

    def bert_input_preprocess(self,
                              examples: Example):
        """
        Minibatch is a list of examples.
        This function converts words to IDs and returns
        torch tensors to be used as input/targets.
        """
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        # pad_token_id = tokenizer.pad_token_id
        cls_token = torch.tensor([cls_token_id])  # .to(device=device)
        sep_token = torch.tensor([sep_token_id])  # .to(device=device)
        inputs, exp_labelses, cls_labels, position_idses, attention_masks, query_masks = [], [], [], [], [], []

        for example in examples:
            cls_label = example.label

            q = example.query
            query_len = len(q)

            max_doc_len = self.max_length - query_len - 2
            d = example.tokens[:max_doc_len]
            exp_labels = example.token_labels[:max_doc_len]
            doc_len = len(d)

            pad_len = self.max_length - query_len - doc_len - 2
            padding = torch.zeros(pad_len).type(torch.int)

            input = torch.cat([cls_token, q, sep_token, d, padding])
            exp_labels = torch.cat([torch.zeros(query_len + 2).type(torch.int),
                                    exp_labels, padding])
            position_ids = torch.cat([torch.arange(0, query_len + 1),
                                      torch.arange(0, doc_len + 1),
                                      padding])
            attention_mask = torch.cat([torch.ones(query_len + doc_len + 2),
                                        padding.type(torch.float)])
            query_mask = torch.cat([torch.ones(query_len + 2),
                                    torch.zeros(doc_len + pad_len)]).type(torch.bool)
            inputs.append(input)
            exp_labelses.append(exp_labels)
            cls_labels.append(cls_label)
            position_idses.append(position_ids)
            attention_masks.append(attention_mask)
            query_masks.append(query_mask)

        return (
            torch.stack(inputs, dim=0),
            torch.stack(exp_labelses, dim=0),
            torch.LongTensor(cls_labels),
            torch.stack(position_idses, dim=0),
            torch.stack(attention_masks, dim=0),
            torch.stack(query_masks, dim=0)
        )

            # if len(q) + len(d) + 2 > max_length:
            #     d = d[:(max_length - len(q) - 2)]
            #     exp_labels = exp_labels[:(max_length - len(q) - 2)]
            # input = torch.cat([cls_token, example.query, sep_token,
            #                    example.tokens[:(max_length - len(q) - 2),
            #                    ]])
            # exp_label = torch.cat([torch.Tensor([0] * (len(example.query) + 2)),
            #                        example.token_labels[:(max_length - len(q) - 2)]])
            # position_ids
            # position_ids.append(torch.tensor(list(range(0, len(q) + 1)) + list(range(0, len(d) + 1))))  # tokens
            #     # positions starts from 1, the two 0s are for [cls] and [sep], [pad]s are also nominated as pos 0
            #
            # inputs = PaddedSequence.autopad(inputs, batch_first=True, padding_value=pad_token_id, device=device)
            # positions = PaddedSequence.autopad(position_ids, batch_first=True, padding_value=0, device=device)
            # exp_labels = PaddedSequence.autopad(exp_labels, batch_first=True, padding_value=0, device=device)
            # attention_masks = inputs.mask(on=1, off=0).type(torch.float).to(device=device)

        # return input, exp_labels, cls_label, position_ids, attention_mask
