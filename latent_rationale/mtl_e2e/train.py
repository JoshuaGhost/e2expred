import json
import wandb
from collections import OrderedDict, defaultdict
import time

import random

import os
import numpy as np

import torch
from pytorch_transformers import AdamW
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

from latent_rationale import eraser_metrics
from latent_rationale.common.util import initialize_model_, get_device, print_parameters, get_minibatch, \
    make_kv_string, write_jsonl
from latent_rationale.common.eraser_utils import convert_to_eraser_json, load_eraser_data
from latent_rationale.mtl_e2e.data import MTLDataLoader
from latent_rationale.mtl_e2e.evaluate import evaluate
from latent_rationale.mtl_e2e.models.expred_e2e import HardKumaE2E
from latent_rationale.mtl_e2e.utils import get_args  # , bert_input_preprocess, numerify_labels, tokenize_query_doc
from latent_rationale.mtl_e2e.predict import predict

device = get_device()
print("device:", device)

verbose = False


def train(model, conf):
    # initialize_model_(model, conf)
    initialize_model_(model)

    optimizer = AdamW(model.parameters(), lr=conf["lr"], weight_decay=conf["weight_decay"])

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=conf["lr_decay"], patience=conf["patience"],
        verbose=True, cooldown=conf["cooldown"], threshold=conf["improvement_threshold"],
        min_lr=conf["min_lr"])

    iter_i = 0
    epoch_train_loss = 0.
    epoch_optional_loss = defaultdict(float)
    start = time.time()
    losses = []
    metrices_names = "cls_acc exp_f1 best_eval".split()
    metrics = {name: [] for name in metrices_names}
    best_eval_iter = 0
    best_eval_loss = 1.0e9
    weights = conf['weights']
    tolerance = conf['tolerance']

    model = model.to(device)

    resume_path = os.path.join(save_path, "model_resume.pt")
    if conf['resume_snapshot'] and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path)
        model.load_state_dict(ckpt["state_dict"])
        conf = ckpt['cfg']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        best_eval_loss = ckpt['best_eval']
        best_eval_iter = ckpt['best_iter']
        iter_i = ckpt['iter_i']
        metrics = ckpt['metrics']
        losses = ckpt['losses']

    best_model_name = os.path.join(save_path, "best_model.pt")

    while True:
        for batch in train_data:
            # for batch in get_minibatch(train_data, batch_size=batch_size, shuffle=True):
            # done training
            if iter_i == iter_total:
                print("# Done training, loading the best model")
                if os.path.exists(best_model_name):
                    ckpt = torch.load(best_model_name)
                    model.load_state_dict(ckpt["state_dict"])
                else:
                    print("# No model found.")

                print("# Evaluating")
                final_dev_eval = evaluate(
                    model, dev_data, tokenizer, weights, label_id_to_name,
                    batch_size=eval_batch_size,
                    max_length=max_length, device=device, tolerance=tolerance)
                test_eval = evaluate(
                    model, test_data, tokenizer, weights, label_id_to_name,
                    batch_size=eval_batch_size,
                    max_length=max_length, device=device, tolerance=tolerance)

                wandb.log({'best_eval_iter': best_eval_iter,
                           'final_dev': final_dev_eval,
                           'test': test_eval})
                print(f"best model iter {best_eval_iter}: "
                      f"final_dev {make_kv_string(final_dev_eval)} "
                      f"test {make_kv_string(test_eval)}")

                # save result
                result_path = os.path.join(save_path, "results.json")

                conf["best_iter"] = best_eval_iter

                for k, v in dev_eval.items():
                    conf["dev_" + k] = v
                    # writer.add_scalar('best/dev/' + k, v, iter_i)

                for k, v in test_eval.items():
                    print("test", k, v)
                    conf["test_" + k] = v
                    # writer.add_scalar('best/test/' + k, v, iter_i)

                # writer.close()

                with open(result_path, mode="w") as f:
                    json.dump(conf, f, indent=4)

                if decode_split == 'train':
                    data, orig_data = train_data, orig_train_data
                elif decode_split == 'dev':
                    data, orig_data = dev_data, orig_dev_data
                elif decode_split == 'test':
                    data, orig_data = test_data, orig_test_data
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
                wandb.save(os.path.join(save_path, f'{decode_split}_decoded.jsonl'))
                return losses, metrics

            # the default collate_fn collates a list of such tuples into a single tuple of a batched image tensor and a
            # batched class label Tensor. cite: https://pytorch.org/docs/stable/data.html#map-style-datasets
            inputs, exp_labels, cls_labels, positions, attention_masks, query_masks = batch

            exp_labels = exp_labels.cuda()
            cls_labels = cls_labels.cuda()

            epoch = iter_i // iters_per_epoch
            model.train()
            aux_pred_p, cls_pred_p, soft_exp_pred, hard_exp_pred = model(inputs=inputs,
                                                                         attention_masks=attention_masks,
                                                                         positions=positions,
                                                                         query_masks=query_masks)
            loss, loss_optional = model.get_loss(aux_pred_p, cls_pred_p, cls_labels,
                                                 soft_exp_pred, exp_labels.data,
                                                 attention_masks, weights, tolerance=tolerance, epoch=epoch)
            # wandb.log({'training':{'loss':loss,
            #                        'loss_optional': loss_optional}})
            epoch_train_loss += loss.item()
            for k, v in loss_optional.items():
                epoch_optional_loss[k] += v

            model.zero_grad()  # erase previous gradients
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=conf["max_grad_norm"])
            optimizer.step()

            iter_i += 1

            # print info
            if iter_i % conf["print_every"] == 0:
                epoch_train_loss = epoch_train_loss / conf["print_every"]
                # writer.add_scalar('train/loss', train_loss, iter_i)
                wandb.log({'train': {'loss': epoch_train_loss,
                                     'iter_i': iter_i}})
                epoch_optional_loss = {k: v / conf['print_every'] for k, v in epoch_optional_loss.items()}
                wandb.log(
                    {
                        'train': {
                            'loss': epoch_train_loss,
                            'optional': epoch_optional_loss,
                            'iter_i': iter_i,
                            'epoch': epoch
                        }
                    }
                )
                # writer.add_scalar('train/' + k, v, iter_i)
                print_str = make_kv_string(loss_optional)
                min_elapsed = (time.time() - start) // 60
                print(f"Epoch {epoch} Iter {iter_i} time={min_elapsed}m loss={epoch_train_loss:0.4} {print_str}")
                losses.append(epoch_train_loss)
                epoch_train_loss = 0.
                epoch_optional_loss = defaultdict(float)

            # evaluate
            if iter_i % eval_every == 0:
                dev_eval = evaluate(
                    model, dev_data, tokenizer, weights, label_id_to_name,
                    batch_size=eval_batch_size,
                    max_length=max_length, device=device
                )

                for name in metrices_names[:-1]:  # no 'best_eval' in the evaluation results
                    metrics[name].append(dev_eval[name])
                # for k, v in dev_eval.items():
                #     writer.add_scalar('dev/' + k, v, iter_i)
                wandb.log({'eval': dev_eval,
                           'iter_i': iter_i,
                           'epoch': epoch})

                print(f"# epoch {epoch} iter {iter_i}: dev {make_kv_string(dev_eval)}")

                # save best model parameters
                compare_score = dev_eval.get('obj', dev_eval['loss'])
                scheduler.step(compare_score)  # adjust learning rate

                if (compare_score < (best_eval_loss * (1 - conf["improvement_threshold"]))) and \
                        iter_i > (3 * iters_per_epoch):
                    print(f"***highscore*** {compare_score:0.4}")
                    best_eval_loss = compare_score
                    best_eval_iter = iter_i

                    wandb.log({'best_eval': dev_eval})
                    # for k, v in dev_eval.items():
                    #     writer.add_scalar('best/dev/' + k, v, iter_i)

                    os.makedirs(save_path, exist_ok=True)

                    ckpt = {
                        "state_dict": model.state_dict(),
                        "cfg": conf,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval": best_eval_loss,
                        "best_iter": best_eval_iter
                    }
                    torch.save(ckpt, best_model_name)

            # save checkpoint
            if iter_i % save_every == 0:
                ckpt = {
                    "state_dict": model.state_dict(),
                    "cfg": conf,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_eval": best_eval_loss,
                    "best_iter": best_eval_iter,
                    "iter_i": iter_i,
                    "metrics": metrics,
                    "losses": losses
                }
                torch.save(ckpt, resume_path)


if __name__ == "__main__":
    torch.manual_seed(12345678)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(87654321)
    random.seed(32767)

    training_conf, model_conf = get_args()

    dataset_name = training_conf['dataset_name']
    unit_test = training_conf.get('unit_test', False)

    wandb.init(name=f'e2expred on {dataset_name} {"unit_test " if unit_test else ""}'
                    f'(GECO l0={model_conf.weights["selection"]}, kappa={training_conf["tolerance"]})',
               entity='explainable-nlp',
               project=f'e2expred{"-unit_test" if unit_test else ""}')
    wandb.config.update(training_conf)

    save_path = os.path.join(training_conf['save_path'], dataset_name)
    os.makedirs(save_path, exist_ok=True)

    label_id_to_name = training_conf['classes']
    decode_split = training_conf['decode_split']
    epochs_total = training_conf['epochs_total']
    batch_size = training_conf["batch_size"]
    eval_batch_size = training_conf.get("eval_batch_size", batch_size)
    label_name_to_id = OrderedDict({k: v for v, k in enumerate(label_id_to_name)})
    max_length = min(training_conf['mtl']['max_length'], training_conf['cls']['max_length'])
    data_dir = os.path.join(training_conf['data_dir'], dataset_name)
    training_conf['model_common']['num_labels'] = len(label_id_to_name)
    merge_evidences = training_conf.get('merge_evidences', False)
    # train_on_part = conf['train_on_part']

    tokenizer = BertTokenizer.from_pretrained(training_conf['bert_vocab'])

    orig_train_data, orig_dev_data, orig_test_data = load_eraser_data(data_dir, merge_evidences)
    cache_dir_train = os.path.join(save_path, "train_preprocessed.pkl")
    train_data = MTLDataLoader(orig_train_data, label_name_to_id, tokenizer, max_length,
                               batch_size, shuffle=True, num_workers=18,
                               cache_fname=cache_dir_train)
    cache_dir_dev = os.path.join(save_path, "dev_preprocessed.pkl")
    dev_data = MTLDataLoader(orig_dev_data, label_name_to_id, tokenizer, max_length,
                             batch_size, shuffle=False, num_workers=18,
                             cache_fname=cache_dir_dev)
    cache_dir_test = os.path.join(save_path, "test_preprocessed.pkl")
    test_data = MTLDataLoader(orig_test_data, label_name_to_id, tokenizer, max_length,
                              batch_size, shuffle=False, num_workers=18,
                              cache_fname=cache_dir_test)

    print("#train: ", len(train_data))
    print("#dev: ", len(dev_data))
    print("#test: ", len(orig_test_data))

    # epochs_total, iters_per_epoch, iter_total = cal_train_confs(train_data, conf)

    iters_per_epoch = len(train_data) // training_conf['batch_size']
    iter_total = iters_per_epoch * epochs_total
    print("Set iter_total to {}".format(iter_total))

    save_every = max(iters_per_epoch, training_conf['save_every'])
    print(f"Set save_every to {save_every}")

    eval_every = max(iters_per_epoch, training_conf['eval_every'])
    print("Set eval_every to {}".format(eval_every))

    if verbose:
        example = dev_data[0]
        print("First train example:", example)
        print("First train example tokens:", example.tokens)
        print("First train example label:", example.label)

    model_ids = list(map(int, filter(lambda x: x.isnumeric(), os.listdir(save_path))))
    if training_conf['resume_snapshot']:
        model_id = str(max(model_ids) if len(model_ids) > 0 else 0)
    else:
        model_id = str(max(model_ids) + 1 if len(model_ids) > 0 else 0)
    wandb.config.update({'model_id': model_id})
    save_path = os.path.join(save_path, model_id)
    os.makedirs(save_path, exist_ok=True)
    # cache_dir = os.path.join(save_path, "preprocessed.pkl")
    print(f"model saved under {os.path.abspath(save_path)}")
    with open(os.path.join(save_path, 'config'), 'w+') as fout:
        fout.write(f'training config:\n{str(training_conf)}\n')
        fout.write(f'model config:\n{str(model_conf)}\n')

    # writer = SummaryWriter(log_dir=save_path)  # TensorBoard

    model = HardKumaE2E.new(model_conf, tokenizer, epochs_total)

    train(model, training_conf)
    scores = eraser_metrics.main(
        [
            "--data_dir",
            data_dir,
            "--split",
            "test",
            "--results",
            os.path.join(save_path, "test_decoded.jsonl"),
            "--score_file",
            os.path.join(save_path, "test_scores.json"),
        ]
    )
    wandb.log(scores)
    wandb.save(os.path.join(save_path, "*.json"))
