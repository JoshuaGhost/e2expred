import json
import pickle
from collections import OrderedDict
import time

import random

import os
import numpy as np

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from latent_rationale.common.util import initialize_model_, get_device, print_parameters, get_minibatch, \
    make_kv_string, write_jsonl
from latent_rationale.common.eraser_utils import convert_to_eraser_json, load_eraser_data
from latent_rationale.mtl_e2e.evaluate import evaluate
from latent_rationale.mtl_e2e.models.expred_e2e import HardKumaE2E
from latent_rationale.mtl_e2e.utils import get_args, prepare_minibatch, numerify_labels, tokenize_query_doc
from latent_rationale.mtl_e2e.predict import predict

device = get_device()
print("device:", device)

verbose = False


def train():
    torch.manual_seed(12345678)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(87654321)
    random.seed(32767)
    # torch.manual_seed(114514)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(114514)
    # random.seed(114514)

    conf, model_conf = get_args()

    print(conf)

    batch_size = conf["batch_size"]
    eval_batch_size = conf.get("eval_batch_size", batch_size)

    dataset_name = conf['dataset_name']
    tokenizer = BertTokenizer.from_pretrained(conf['bert_vocab'])
    label_id_to_name = conf['classes']
    label_name_to_id = OrderedDict({k: v for v, k in enumerate(label_id_to_name)})
    conf['model_common']['num_labels'] = len(label_id_to_name)

    # train_on_part = conf['train_on_part']
    decode_split = conf['decode_split']

    print("Loading data")
    if conf['lambda_init'] > 0.:
        l0 = f"_{round(conf['weights']['selection'] * 100)}pct"
    else:
        l0 = '_no_l0'
    ch = conf['weights'].get('lasso', 0.)
    if ch > 0.:
        lasso = f'_lasso_{ch}'
    else:
        lasso = '_no_coherence'

    def get_modelname_component(conf, entry, entry_as_comp=False):
        value = conf.get(entry, None)
        if value is None:
            ret = ''
        elif isinstance(value, str):
            if entry_as_comp:
                ret = '_' + entry
            else:
                ret = '_' + value
        elif isinstance(value, bool):
            if value:
                ret = '_' + entry
            else:
                ret = ''
        elif isinstance(value, int) or isinstance(value, float):
            ret = f'_{entry}_{value}'
        return ret

    epochs_total = conf['epochs_total']

    weights_scheduler = get_modelname_component(conf, 'weights_scheduler')
    share_encoder = get_modelname_component(conf, 'share_encoder')
    warm_start_mtl = get_modelname_component(conf, 'warm_start_mtl', True)
    warm_start_cls = get_modelname_component(conf, 'warm_start_cls', True)
    exp_threshold = get_modelname_component(conf, 'exp_threshold')
    soft_selection = get_modelname_component(conf, 'soft_selection', True)
    num_epochs = f"_epochs_{epochs_total}"

    w_aux = conf['weights'].get('w_aux', None)
    if w_aux is not None:
        w_exp = conf['weights']['w_exp']
    else:
        w_aux = 1.
        w_exp = 1.
        conf['weights']['w_aux'] = 1.
        conf['weights']['w_exp'] = 1.
    ws_mtl = f"_waux_{w_aux}_wexp_{w_exp}"

    save_path = conf['save_path'] + \
                f"/{dataset_name}/mtl_e2e" + \
                l0 + lasso + \
                weights_scheduler + \
                ws_mtl + \
                share_encoder + \
                warm_start_mtl + \
                warm_start_cls + \
                exp_threshold + \
                soft_selection + \
                num_epochs
    print(os.path.abspath(save_path))
    data_dir = conf['data_dir']
    data_dir = os.path.join(data_dir, dataset_name)
    cache_dir = os.path.join(save_path, "preprocessed.pkl")
    if os.path.isfile(cache_dir):
        print(f'Preprocessed dataset found at {cache_dir}, loading...')
        with open(cache_dir, 'rb') as fin:
            train_data, dev_data, test_data, \
            orig_train_data, orig_dev_data, orig_test_data = pickle.load(fin)
    else:
        print("Preprocessing the dataset for the first time.")
        os.makedirs(save_path, exist_ok=True)
        merge_evidences = bool(conf.get('merge_evidences', 0))
        orig_train_data, orig_dev_data, orig_test_data = load_eraser_data(data_dir, merge_evidences)
        train_data, dev_data, test_data = [numerify_labels(dataset, label_name_to_id)
                                           for dataset in [orig_train_data, orig_dev_data, orig_test_data]]

        train_data, dev_data, test_data = [[tokenize_query_doc(example, tokenizer) for example in dataset]
                                           for dataset in [train_data, dev_data, test_data]]
        with open(cache_dir, "wb+") as fout:
            pickle.dump((train_data, dev_data, test_data,
                         orig_train_data, orig_dev_data, orig_test_data), fout)
        print(f'preprocessed dataset dumped at {cache_dir}')

    print("#train: ", len(train_data))
    print("#dev: ", len(dev_data))
    print("#test: ", len(orig_test_data))

    # epochs_total, iters_per_epoch, num_iterations = cal_train_confs(train_data, conf)

    iters_per_epoch = len(train_data) // conf['batch_size']

    num_iterations = iters_per_epoch * epochs_total
    print("Set num_iterations to {}".format(num_iterations))

    save_every = max(iters_per_epoch, conf['save_every'])
    print(f"Set save_every to {save_every}")

    eval_every = max(iters_per_epoch, conf['eval_every'])
    print("Set eval_every to {}".format(eval_every))

    if verbose:
        example = dev_data[0]
        print("First train example:", example)
        print("First train example tokens:", example.tokens)
        print("First train example label:", example.label)

    writer = SummaryWriter(log_dir=save_path)  # TensorBoard

    # Build model

    model = HardKumaE2E.new(conf, tokenizer, epochs_total)
    initialize_model_(model)

    optimizer = Adam(model.parameters(), lr=conf["lr"], weight_decay=conf["weight_decay"])

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=conf["lr_decay"], patience=conf["patience"],
        verbose=True, cooldown=conf["cooldown"], threshold=conf["improvement_threshold"],
        min_lr=conf["min_lr"])

    iter_i = 0
    train_loss = 0.
    start = time.time()
    losses = []
    metrices_names = "cls_acc cls_f1 exp_precision exp_recall exp_f1 best_eval".split()
    metrics = {name: [] for name in metrices_names}
    best_iter = 0
    best_eval = 1.0e9
    weights = conf['weights']
    max_length = min(conf['mtl']['max_length'], conf['cls']['max_length'])

    model = model.to(device)

    resume_path = os.path.join(save_path, "model_resume.pt")
    if conf['resume_snapshot'] and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path)
        model.load_state_dict(ckpt["state_dict"])
        conf = ckpt['cfg']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        best_eval = ckpt['best_eval']
        best_iter = ckpt['best_iter']
        iter_i = ckpt['iter_i']
        metrics = ckpt['metrics']
        losses = ckpt['losses']

    while True:  # when we run out of examples, shuffle and continue
        for batch in get_minibatch(train_data, batch_size=batch_size,
                                   shuffle=True, train_on_part=conf['train_on_part']):
            # for batch in get_minibatch(train_data, batch_size=batch_size, shuffle=True):
            # done training
            if iter_i == num_iterations:
                print("# Done training")
                # evaluate on test with best model
                print("# Loading the best model")
                path = os.path.join(save_path, "model.pt")
                if os.path.exists(path):
                    ckpt = torch.load(path)
                    model.load_state_dict(ckpt["state_dict"])
                else:
                    print("No model found.")

                print("# Evaluating")
                dev_eval = evaluate(
                    model, dev_data, tokenizer, weights, label_id_to_name,
                    batch_size=eval_batch_size,
                    max_length=max_length, device=device)
                test_eval = evaluate(
                    model, test_data, tokenizer, weights, label_id_to_name,
                    batch_size=eval_batch_size,
                    max_length=max_length, device=device)

                print(f"best model iter {best_iter}: "
                      f"dev {make_kv_string(dev_eval)} "
                      f"test {make_kv_string(test_eval)}")

                # save result
                result_path = os.path.join(save_path, "results.json")

                conf["best_iter"] = best_iter

                for k, v in dev_eval.items():
                    conf["dev_" + k] = v
                    writer.add_scalar('best/dev/' + k, v, iter_i)

                for k, v in test_eval.items():
                    print("test", k, v)
                    conf["test_" + k] = v
                    writer.add_scalar('best/test/' + k, v, iter_i)

                writer.close()

                with open(result_path, mode="w") as f:
                    json.dump(conf, f)

                if decode_split == 'train':
                    data = train_data
                    orig_data = orig_train_data
                elif decode_split == 'dev':
                    data = dev_data
                    orig_data = orig_dev_data
                elif decode_split == 'test':
                    data = test_data
                    orig_data = orig_test_data
                cls_pred_p, soft_exp_pred, hard_exp_pred = predict(
                    model, data, tokenizer,
                    batch_size=eval_batch_size,
                    max_length=512, device=device
                )
                test_decoded = [convert_to_eraser_json(p_cls_p, soft_exp_p, hard_exp_p, ot, tokenizer, label_id_to_name)
                                for p_cls_p, soft_exp_p, hard_exp_p, ot in zip(cls_pred_p,
                                                                               soft_exp_pred,
                                                                               hard_exp_pred,
                                                                               orig_data)]
                write_jsonl(test_decoded, os.path.join(save_path, f'{decode_split}_decoded.jsonl'))
                # cls_pred_p, soft_exp_pred, hard_exp_pred = predict(
                #     model, test_data, tokenizer,
                #     batch_size=eval_batch_size,
                #     max_length=512, device=device
                # )
                # test_decoded = [convert_to_eraser_json(p_cls_p, soft_exp_p, hard_exp_p, ot, tokenizer, i2t)
                #                 for p_cls_p, soft_exp_p, hard_exp_p, ot in zip(cls_pred_p,
                #                                                                soft_exp_pred,
                #                                                                hard_exp_pred,
                #                                                                orig_test_data)]
                # write_jsonl(test_decoded, os.path.join(save_path, f'test_decoded.jsonl'))
                return losses, metrics

            epoch = iter_i // iters_per_epoch
            model.train()
            inputs, exp_labels, labels, positions, attention_masks, padding_masks = prepare_minibatch(batch,
                                                                                                      tokenizer=tokenizer,
                                                                                                      max_length=max_length,
                                                                                                      device=device)

            aux_pred_p, cls_pred_p, soft_exp_pred, hard_exp_pred = model(inputs.data,
                                                                         attention_masks=attention_masks,
                                                                         padding_masks=padding_masks,
                                                                         positions=positions.data)

            loss, loss_optional = model.get_loss(aux_pred_p, cls_pred_p, labels, soft_exp_pred, exp_labels.data,
                                                 padding_masks, weights, epoch)
            model.zero_grad()  # erase previous gradients

            train_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=conf["max_grad_norm"])
            optimizer.step()

            iter_i += 1

            # print info
            if iter_i % conf["print_every"] == 0:
                train_loss = train_loss / conf["print_every"]
                writer.add_scalar('train/loss', train_loss, iter_i)
                for k, v in loss_optional.items():
                    writer.add_scalar('train/' + k, v, iter_i)
                print_str = make_kv_string(loss_optional)
                min_elapsed = (time.time() - start) // 60
                print(f"Epoch {epoch} Iter {iter_i} time={min_elapsed}m loss={train_loss:0.4} {print_str}")
                losses.append(train_loss)
                train_loss = 0.

            # evaluate
            if iter_i % eval_every == 0:
                dev_eval = evaluate(
                    model, dev_data, tokenizer, weights, label_id_to_name,
                    batch_size=eval_batch_size,
                    max_length=max_length, device=device
                )

                for name in metrices_names[:-1]:  # no 'best_eval' in the evaluation results
                    metrics[name].append(dev_eval[name])
                for k, v in dev_eval.items():
                    writer.add_scalar('dev/' + k, v, iter_i)

                print(f"# epoch {epoch} iter {iter_i}: dev {make_kv_string(dev_eval)}")

                # save best model parameters
                compare_score = dev_eval["loss"]
                if "obj" in dev_eval:
                    compare_score = dev_eval["obj"]

                scheduler.step(compare_score)  # adjust learning rate

                if (compare_score < (best_eval * (1 - conf["improvement_threshold"]))) and \
                        iter_i > (3 * iters_per_epoch):
                    print(f"***highscore*** {compare_score:0.4}")
                    best_eval = compare_score
                    best_iter = iter_i

                    for k, v in dev_eval.items():
                        writer.add_scalar('best/dev/' + k, v, iter_i)

                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    ckpt = {
                        "state_dict": model.state_dict(),
                        "cfg": conf,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval": best_eval,
                        "best_iter": best_iter
                    }
                    path = os.path.join(save_path, "model.pt")
                    torch.save(ckpt, path)

            # save checkpoint
            if iter_i % save_every == 0:
                ckpt = {
                    "state_dict": model.state_dict(),
                    "cfg": conf,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_eval": best_eval,
                    "best_iter": best_iter,
                    "iter_i": iter_i,
                    "metrics": metrics,
                    "losses": losses
                }
                torch.save(ckpt, resume_path)


if __name__ == "__main__":
    train()
