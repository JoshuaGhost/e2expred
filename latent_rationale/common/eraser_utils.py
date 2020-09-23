import json
from collections import defaultdict

from copy import deepcopy
from dataclasses import dataclass, asdict, is_dataclass
from typing import Dict, List, Set, Tuple, Union, FrozenSet

import os

from itertools import chain

import numpy as np

from .util import Example


@dataclass(eq=True, frozen=True)
class Evidence:
    """
    (docid, start_token, end_token) form the only official Evidence; sentence level annotations are for convenience.
    Args:
        text: Some representation of the evidence text
        docid: Some identifier for the document
        start_token: The canonical start token, inclusive
        end_token: The canonical end token, exclusive
        start_sentence: Best guess start sentence, inclusive
        end_sentence: Best guess end sentence, exclusive
    """
    text: Union[str, Tuple[int], Tuple[str]]
    docid: str
    start_token: int = -1
    end_token: int = -1
    start_sentence: int = -1
    end_sentence: int = -1


@dataclass(eq=True, frozen=True)
class Annotation:
    """
    Args:
        annotation_id: unique ID for this annotation element
        query: some representation of a query string
        evidences: a set of "evidence groups".
            Each evidence group is:
                * sufficient to respond to the query (or justify an answer)
                * composed of one or more Evidences
                * may have multiple documents in it (depending on the dataset)
                    - e-snli has multiple documents
                    - other datasets do not
        classification: str
        query_type: Optional str, additional information about the query
        docids: a set of docids in which one may find evidence.
    """
    annotation_id: str
    query: Union[str, Tuple[int]]
    evidences: Union[Set[Tuple[Evidence]], FrozenSet[Tuple[Evidence]]]
    classification: str
    query_type: str = None
    docids: Set[str] = None

    def all_evidences(self) -> Tuple[Evidence]:
        return tuple(list(chain.from_iterable(self.evidences)))

# from preprocessing import convert_bert_features
# from bert_rational_feature import InputRationalExample, convert_examples_to_features
# from eraser_utils import extract_doc_ids_from_annotations

# from utils import convert_subtoken_ids_to_tokens


# def remove_rations(sentence, annotation):
#     sentence = sentence.lower().split()
#     rationales = annotation['rationales'][0]['hard_rationale_predictions']
#     rationales = [{'end_token': 0, 'start_token': 0}] \
#                  + sorted(rationales, key=lambda x: x['start_token']) \
#                  + [{'start_token': len(sentence), 'end_token': len(sentence)}]
#     ret = []
#     for rat_id, rat in enumerate(rationales[:-1]):
#         ret += ['.'] * (rat['end_token'] - rat['start_token']) \
#                + sentence[rat['end_token']
#                           : rationales[rat_id + 1]['start_token']]
#     return ' '.join(ret)


# def extract_rations(sentence, rationale):
#     sentence = sentence.lower().split()
#     rationales = rationale['rationales'][0]['hard_rationale_predictions']
#     rationales = [{'end_token': 0, 'start_token': 0}] \
#                  + sorted(rationales, key=lambda x: x['start_token']) \
#                  + [{'start_token': len(sentence), 'end_token': len(sentence)}]
#     ret = []
#     for rat_id, rat in enumerate(rationales[:-1]):
#         ret += sentence[rat['start_token']: rat['end_token']] \
#                + ['.'] * (rationales[rat_id + 1]
#                           ['start_token'] - rat['end_token'])
#     return ' '.join(ret)
#
#
# def ce_load_bert_features(rationales, docs, label_list, decorate, max_seq_length, gpu_id, tokenizer=None):
#     input_examples = []
#     for r_idx, rational in enumerate(rationales):
#         text_a = rational['query']
#         docids = rational['docids']
#         sentences = chain.from_iterable(docs[docid] for docid in docids)
#         flattened_tokens = chain(*sentences)
#         text_b = ' '.join(flattened_tokens)
#         text_b = decorate(text_b, rational)
#         label = rational['classification']
#         evidences = None
#         input_examples.append(InputRationalExample(guid=None,
#                                                    text_a=text_a,
#                                                    text_b=text_b,
#                                                    label=label,
#                                                    evidences=evidences))
#     features = convert_examples_to_features(input_examples, label_list, max_seq_length, tokenizer)
#     return features
#
#
# def ce_preprocess(rationales, docs, label_list, dataset_name, decorate, max_seq_length, exp_output, gpu_id, tokenizer):
#     features = ce_load_bert_features(rationales, docs, label_list, decorate, max_seq_length, gpu_id, tokenizer)
#
#     with_rations = ('cls' not in dataset_name)
#     with_lable_id = ('seq' not in dataset_name)
#
#     return convert_bert_features(features, with_lable_id, with_rations, exp_output)
#
#
# def get_cls_score(model, rationales, docs, label_list, dataset, decorate, max_seq_length, exp_output, gpu_id, tokenizer):
#     rets = ce_preprocess(rationales, docs, label_list, dataset, decorate, max_seq_length, exp_output, gpu_id, tokenizer)
#     _input_ids, _input_masks, _segment_ids, _rations, _labels = rets
#
#     _inputs = [_input_ids, _input_masks, _segment_ids]
#     _pred = model.predict(_inputs)
#     return (np.hstack([1 - _pred[0], _pred[0]]))
#
#
# def add_cls_scores(res, cls, c, s, label_list):
#     res['classification_scores'] = {label_list[0]: cls[0], label_list[1]: cls[1]}
#     res['comprehensiveness_classification_scores'] = {label_list[0]: c[0], label_list[1]: c[1]}
#     res['sufficiency_classification_scores'] = {label_list[0]: s[0], label_list[1]: s[1]}
#     return res


def pred_to_exp_mask(exp_pred, count, threshold):
    if count is None:
        return (np.array(exp_pred).astype(np.float) >= threshold).astype(np.int32)
    temp = [(i, p) for i, p in enumerate(exp_pred)]
    temp = sorted(temp, key=lambda x: x[1], reverse=True)
    ret = np.zeros_like(exp_pred).astype(np.int32)
    for i, _ in temp[:count]:
        ret[i] = 1
    return ret

#
# def rational_bits_to_ev_generator(token_list, raw_input_or_docid, exp_pred, hard_selection_count=None,
#                                   hard_selection_threshold=0.5):
#     in_rationale = False
#     if not isinstance(raw_input_or_docid, Annotation):
#         docid = raw_input_or_docid
#     else:
#         docid = list(extract_doc_ids_from_annotations([raw_input_or_docid]))[0]
#     ev = {'docid': docid,
#           'start_token': -1, 'end_token': -1, 'text': ''}
#     exp_masks = pred_to_exp_mask(
#         exp_pred, hard_selection_count, hard_selection_threshold)
#     for i, p in enumerate(exp_masks):
#         if p == 0 and in_rationale:  # leave rational zone
#             in_rationale = False
#             ev['end_token'] = i
#             ev['text'] = ' '.join(
#                 token_list[ev['start_token']: ev['end_token']])
#             yield deepcopy(ev)
#         elif p == 1 and not in_rationale:  # enter rational zone
#             in_rationale = True
#             ev['start_token'] = i
#     if in_rationale:  # the final non-padding token is rational
#         ev['end_token'] = len(exp_pred)
#         ev['text'] = ' '.join(token_list[ev['start_token']: ev['end_token']])
#         yield deepcopy(ev)
#
#
# # [SEP] == 102
# # [CLS] == 101
# # [PAD] == 0
# def extract_texts(tokens, exps=None, text_a=True, text_b=False):
#     if tokens[0] == 101:
#         endp_text_a = tokens.index(102)
#         if text_b:
#             endp_text_b = endp_text_a + 1 + \
#                           tokens[endp_text_a + 1:].index(102)
#     else:
#         endp_text_a = tokens.index('[SEP]')
#         if text_b:
#             endp_text_b = endp_text_a + 1 + \
#                           tokens[endp_text_a + 1:].index('[SEP]')
#     ret_token = []
#     if text_a:
#         ret_token += tokens[1: endp_text_a]
#     if text_b:
#         ret_token += tokens[endp_text_a + 1: endp_text_b]
#     if exps is None:
#         return ret_token
#     else:
#         ret_exps = []
#         if text_a:
#             ret_exps += exps[1: endp_text_a]
#         if text_b:
#             ret_exps += exps[endp_text_a + 1: endp_text_b]
#         return ret_token, ret_exps
#
#
# def rnr_matrix_to_rational_mask(rnr_matrix):
#     start_logits, end_logits = rnr_matrix[:, :1], rnr_matrix[:, 1:]
#     starts = np.round(start_logits).reshape((-1, 1))
#     ends = np.triu(end_logits)
#     ends = starts * ends
#     ends_args = np.argmax(ends, axis=1)
#     ends = np.zeros_like(ends)
#     for i in range(len(ends_args)):
#         ends[i, ends_args[i]] = 1
#     ends = starts * ends
#     ends = np.sum(ends, axis=0, keepdims=True)
#     rational_mask = np.cumsum(starts.reshape((1, -1)), axis=1) - np.cumsum(ends, axis=1) + ends
#     return rational_mask
#
#
# def pred_to_results(raw_input, input_ids, pred,
#                     hard_selection_count, hard_selection_threshold,
#                     vocab, docs, label_list,
#                     exp_output):
#     cls_pred, exp_pred = pred
#     if exp_output == 'rnr':
#         exp_pred = rnr_matrix_to_rational_mask(exp_pred)
#     exp_pred = exp_pred.reshape((-1,)).tolist()
#     docid = list(raw_input.evidences)[0][0].docid
#     raw_sentence = ' '.join(list(chain.from_iterable(docs[docid])))
#     raw_sentence = re.sub('\x12', '', raw_sentence)
#     raw_sentence = raw_sentence.lower().split()
#     token_ids, exp_pred = extract_texts(input_ids, exp_pred, text_a=False, text_b=True)
#     token_list, exp_pred = convert_subtoken_ids_to_tokens(token_ids, vocab, exp_pred, raw_sentence)
#     result = {'annotation_id': raw_input.annotation_id, 'query': raw_input.query}
#     ev_groups = []
#     result['docids'] = [docid]
#     result['rationales'] = [{'docid': docid}]
#     for ev in rational_bits_to_ev_generator(token_list, raw_input, exp_pred, hard_selection_count,
#                                             hard_selection_threshold):
#         ev_groups.append(ev)
#     result['rationales'][-1]['hard_rationale_predictions'] = ev_groups
#     if exp_output != 'rnr':
#         result['rationales'][-1]['soft_rationale_predictions'] = exp_pred + [0] * (len(raw_sentence) - len(token_list))
#     result['classification'] = label_list[int(round(cls_pred[0]))]
#     return result


@dataclass(eq=True, frozen=True)
class Evidence:
    """
    (docid, start_token, end_token) form the only official Evidence; sentence level annotations are for convenience.
    Args:
        text: Some representation of the evidence text
        docid: Some identifier for the document
        start_token: The canonical start token, inclusive
        end_token: The canonical end token, exclusive
        start_sentence: Best guess start sentence, inclusive
        end_sentence: Best guess end sentence, exclusive
    """
    text: Union[str, List[int]]
    docid: str
    start_token: int = -1
    end_token: int = -1
    start_sentence: int = -1
    end_sentence: int = -1


@dataclass(eq=True, frozen=True)
class Annotation:
    """
    Args:
        annotation_id: unique ID for this annotation element
        query: some representation of a query string
        evidences: a set of "evidence groups".
            Each evidence group is:
                * sufficient to respond to the query (or justify an answer)
                * composed of one or more Evidences
                * may have multiple documents in it (depending on the dataset)
                    - e-snli has multiple documents
                    - other datasets do not
        classification: str
        query_type: Optional str, additional information about the query
        docids: a set of docids in which one may find evidence.
    """
    annotation_id: str
    query: Union[str, List[int]]
    evidences: Set[Tuple[Evidence]]
    classification: str
    query_type: str = None
    docids: Set[str] = None

    def all_evidences(self) -> Tuple[Evidence]:
        return tuple(list(chain.from_iterable(self.evidences)))


def annotations_to_jsonl(annotations, output_file):
    with open(output_file, 'w') as of:
        for ann in sorted(annotations, key=lambda x: x.annotation_id):
            as_json = _annotation_to_dict(ann)
            as_str = json.dumps(as_json, sort_keys=True)
            of.write(as_str)
            of.write('\n')


def _annotation_to_dict(dc):
    # convenience method
    if is_dataclass(dc):
        d = asdict(dc)
        ret = dict()
        for k, v in d.items():
            ret[k] = _annotation_to_dict(v)
        return ret
    elif isinstance(dc, dict):
        ret = dict()
        for k, v in dc.items():
            k = _annotation_to_dict(k)
            v = _annotation_to_dict(v)
            ret[k] = v
        return ret
    elif isinstance(dc, str):
        return dc
    elif isinstance(dc, (set, frozenset, list, tuple)):
        ret = []
        for x in dc:
            ret.append(_annotation_to_dict(x))
        return tuple(ret)
    else:
        return dc


def load_jsonl(fp: str) -> List[dict]:
    ret = []
    with open(fp, 'r') as inf:
        for line in inf:
            content = json.loads(line)
            ret.append(content)
    return ret


# def annotations_from_jsonl(fp: str) -> List[Annotation]:
def annotations_from_jsonl(fp: str):
    ret = []
    with open(fp, 'r') as inf:
        for line in inf:
            content = json.loads(line)
            ev_groups = []
            for ev_group in content['evidences']:
                ev_group = tuple([Evidence(**ev) for ev in ev_group])
                ev_groups.append(ev_group)
            content['evidences'] = frozenset(ev_groups)
            ret.append(Annotation(**content))
    return ret


def load_datasets(data_dir: str) -> Tuple[List[Annotation], List[Annotation], List[Annotation]]:
    """Loads a training, validation, and test dataset

    Each dataset is assumed to have been serialized by annotations_to_jsonl,
    that is it is a list of json-serialized Annotation instances.
    """
    train_data = annotations_from_jsonl(os.path.join(data_dir, 'train.jsonl'))
    val_data = annotations_from_jsonl(os.path.join(data_dir, 'val.jsonl'))
    test_data = annotations_from_jsonl(os.path.join(data_dir, 'test.jsonl'))
    return train_data, val_data, test_data


def load_documents(data_dir: str, docids: Set[str] = None) -> Dict[str, List[List[str]]]:
    """Loads a subset of available documents from disk.

    Each document is assumed to be serialized as newline ('\n') separated sentences.
    Each sentence is assumed to be space (' ') joined tokens.
    """
    if os.path.exists(os.path.join(data_dir, 'docs.jsonl')):
        assert not os.path.exists(os.path.join(data_dir, 'docs'))
        return load_documents_from_file(data_dir, docids)

    docs_dir = os.path.join(data_dir, 'docs')
    res = dict()
    if docids is None:
        docids = sorted(os.listdir(docs_dir))
    else:
        docids = sorted(set(str(d) for d in docids))
    for d in docids:
        with open(os.path.join(docs_dir, d), 'r') as inf:
            lines = [l.strip() for l in inf.readlines()]
            lines = list(filter(lambda x: bool(len(x)), lines))
            tokenized = [list(filter(lambda x: bool(len(x)), line.strip().split(' '))) for line in lines]
            res[d] = tokenized
    return res


def load_flattened_documents(data_dir: str, docids: Set[str]) -> Dict[str, List[str]]:
    """Loads a subset of available documents from disk.

    Returns a tokenized version of the document.
    """
    unflattened_docs = load_documents(data_dir, docids)
    flattened_docs = dict()
    for doc, unflattened in unflattened_docs.items():
        flattened_docs[doc] = list(chain.from_iterable(unflattened))
    return flattened_docs


def intern_documents(documents: Dict[str, List[List[str]]], word_interner: Dict[str, int], unk_token: str):
    """
    Replaces every word with its index in an embeddings file.

    If a word is not found, uses the unk_token instead
    """
    ret = dict()
    unk = word_interner[unk_token]
    for docid, sentences in documents.items():
        ret[docid] = [[word_interner.get(w, unk) for w in s] for s in sentences]
    return ret


def intern_annotations(annotations: List[Annotation], word_interner: Dict[str, int], unk_token: str):
    ret = []
    for ann in annotations:
        ev_groups = []
        for ev_group in ann.evidences:
            evs = []
            for ev in ev_group:
                evs.append(Evidence(
                    text=tuple([word_interner.get(t, word_interner[unk_token]) for t in ev.text.split()]),
                    docid=ev.docid,
                    start_token=ev.start_token,
                    end_token=ev.end_token,
                    start_sentence=ev.start_sentence,
                    end_sentence=ev.end_sentence))
            ev_groups.append(tuple(evs))
        ret.append(Annotation(annotation_id=ann.annotation_id,
                              query=tuple([word_interner.get(t, word_interner[unk_token]) for t in ann.query.split()]),
                              evidences=frozenset(ev_groups),
                              classification=ann.classification,
                              query_type=ann.query_type))
    return ret


def load_documents_from_file(data_dir: str, docids: Set[str] = None) -> Dict[str, List[List[str]]]:
    """Loads a subset of available documents from 'docs.jsonl' file on disk.

    Each document is assumed to be serialized as newline ('\n') separated sentences.
    Each sentence is assumed to be space (' ') joined tokens.
    """
    docs_file = os.path.join(data_dir, 'docs.jsonl')
    documents = load_jsonl(docs_file)
    documents = {doc['docid']: doc['document'] for doc in documents}
    res = dict()
    if docids is None:
        docids = sorted(list(documents.keys()))
    else:
        docids = sorted(set(str(d) for d in docids))
    for d in docids:
        lines = documents[d].split('\n')
        tokenized = [line.strip().split(' ') for line in lines]
        res[d] = tokenized
    return res


def rational_bits_to_ev_generator(token_list, docid, exp_masks, hard_selection_count=None,
                                  hard_selection_threshold=0.5):
    in_rationale = False
    ev = {'docid': docid,
          'start_token': -1, 'end_token': -1, 'text': ''}
    for i, p in enumerate(exp_masks):
        if p == 0 and in_rationale:  # leave rational zone
            in_rationale = False
            ev['end_token'] = i
            ev['text'] = ' '.join(
                token_list[ev['start_token']: ev['end_token']])
            yield deepcopy(ev)
        elif p == 1 and not in_rationale:  # enter rational zone
            in_rationale = True
            ev['start_token'] = i
    if in_rationale:  # the final non-padding token is rational
        ev['end_token'] = len(exp_masks)
        ev['text'] = ' '.join(token_list[ev['start_token']: ev['end_token']])
        yield deepcopy(ev)


def make_eraser_json(cls_p, soft_exp, hard_exp, orig_data, class_labels):
    tokens = orig_data.tokens
    docid = orig_data.docid
    ann_id = orig_data.ann_id
    ev_generator = rational_bits_to_ev_generator(tokens,
                                                 docid,
                                                 hard_exp)
    hard_exp = [ev for ev in ev_generator]

    ret = {
        "annotation_id": ann_id,
        "rationales": [],
        "classification": class_labels[int(cls_p.to('cpu').argmax())],
        "classification_scores": {class_labels[i]: s for i, s in
                                  enumerate(cls_p.to('cpu').tolist())},
        # TODO this should turn into the data distribution for the predicted class
        # "comprehensiveness_classification_scores": 0.0,
        "truth": orig_data.label,
    }
    ret['rationales'].append({
        "docid": docid,
        "hard_rationale_predictions": hard_exp,
        "soft_rationale_predictions": soft_exp,
    })
    #print(ret)
    return ret


def convert_to_eraser_json(cls_pred, soft_exp_pred, hard_exp_pred,
                           orig_data:Example, tokenizer, class_labels):
    st = 0
    soft_exp_final = []
    hard_exp_final = []
    soft_exp_pred = soft_exp_pred.tolist()
    hard_exp_pred = hard_exp_pred.tolist()
    for token in orig_data.tokens:
        if st < len(hard_exp_pred):
            len_token_pieces = len(tokenizer.tokenize(token))
            soft_exp_final.append(max(soft_exp_pred[st : st + len_token_pieces]))
            hard_exp_final.append(max(hard_exp_pred[st: st + len_token_pieces]))
            st += len_token_pieces
        else:
            soft_exp_final.append(0.)
            hard_exp_final.append(0)
    assert len(orig_data.tokens) == len(hard_exp_final)
    assert len(orig_data.tokens) == len(soft_exp_final)
    return make_eraser_json(cls_p=cls_pred,
                            soft_exp=soft_exp_final, hard_exp=hard_exp_final,
                            orig_data=orig_data, class_labels=class_labels)


def token_labels_from_annotation(evs, doc):
    token_labels = np.zeros([len(doc)]).astype(np.int)
    for ev in evs:
        token_labels[ev.start_token: ev.end_token] = 1
    return token_labels


def convert_annotations_to_examples(annotations, docs, merge_evidences):
    examples = []
    for ann in annotations:
        all_evidences = defaultdict(list)
        if merge_evidences:
            for ev in ann.all_evidences():
                all_evidences[ev.docid].append(ev)
            if len(all_evidences) == 0: # posR_101.txt in movies
                continue
            docids = list(all_evidences.keys())
            flattened_docs = [list(chain.from_iterable(docs[docid])) for docid in docids]
            token_labels = [token_labels_from_annotation(all_evidences[docid], doc)
                            for docid, doc in zip(docids, flattened_docs)]
            flattened_tokens = list(chain.from_iterable(flattened_docs))
            flattened_token_labels = list(chain.from_iterable(token_labels))
            label = ann.classification
            query = ann.query
            ann_id = ann.annotation_id
            examples.append(Example(tokens=flattened_tokens,
                                    token_labels=flattened_token_labels,
                                    label=label,
                                    query=query,
                                    ann_id=ann_id,
                                    docid=docids[0])) # TODO: monkey patch warning
    return examples


def load_eraser_data(data_dir: str, merge_evidences: bool) -> Tuple[List[Example], List[Example], List[Example]]:
    train = annotations_from_jsonl(os.path.join(data_dir, 'train.jsonl'))
    val = annotations_from_jsonl(os.path.join(data_dir, 'val.jsonl'))
    test = annotations_from_jsonl(os.path.join(data_dir, 'test.jsonl'))
    docids = set(e.docid for e in chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(train, val, test)))))
    docs = load_documents(data_dir, docids)
    train = convert_annotations_to_examples(train, docs, merge_evidences)
    val = convert_annotations_to_examples(val, docs, merge_evidences)
    test = convert_annotations_to_examples(test, docs, merge_evidences)
    return train, val, test
