import pickle
import os
from torch.utils.data import DataLoader

from latent_rationale.mtl_e2e.utils import tokenize_query_doc, bert_input_preprocess, numerify_label


class MTLDataLoader(DataLoader):

    def _preprocess(self, raw_data, max_length):
        print("Preprocessing the dataset for the first time.")
        data = [numerify_label(ann, self.label_name_to_id) for ann in raw_data]
        # data = numerify_labels(raw_data, self.label_name_to_id)
        data = [tokenize_query_doc(instance, self.tokenizer) for instance in data]
        inputs, exps, labels, positions, attention_masks, padding_masks =\
            bert_input_preprocess(data, self.tokenizer, max_length, device='cpu')
        data = list(zip(inputs.data, exps.data, labels, positions.data, attention_masks.data, padding_masks.data))
        return data

    def __init__(self, raw_data, label_name_to_id, tokenizer, max_length,
                 batch_size, shuffle, num_workers, cache_fname=None):
        self.raw_data = raw_data
        self.cache_fname = cache_fname
        self.label_name_to_id = label_name_to_id
        self.tokenizer = tokenizer
        self.max_length = max_length

        if self.cache_fname is None or not os.path.isfile(self.cache_fname):
            self.data = self._preprocess(self.raw_data, self.max_length)
            if self.cache_fname is not None:
                with open(self.cache_fname, 'wb+') as fout:
                    pickle.dump(self.data, fout)
                print(f'preprocessed dataset dumped at {self.cache_fname}')
        else:
            print(f'Preprocessed dataset found at {self.cache_fname}, loading...')
            with open(self.cache_fname, 'rb') as fin:
                self.data = pickle.load(fin)
        super(MTLDataLoader, self).__init__(self.data, batch_size=batch_size, shuffle=shuffle,
                                            num_workers=num_workers, pin_memory=True)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
