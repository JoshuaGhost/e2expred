import torch
import numpy as np
from sklearn.metrics import classification_report


# def sequence_macro_prf(exp_labels, exp_pred, eps=1e-10):
#     p = ((exp_pred * exp_labels).sum(axis=-1).astype(np.float) /
#          (exp_pred.sum(axis=-1) + eps)).mean() + eps
#     r = ((exp_pred * exp_labels).sum(axis=-1).astype(np.float) /
#          (exp_labels.sum(axis=-1) + eps)).mean() + eps
#     f1 = 2 / (1 / p + 1 / r)
#     return p, r, f1


def cls_prf(labels, preds, label_list, report_prefix):
    if not isinstance(preds[0], int):  # multi-class probabilities
        preds = [int(p.to('cpu').argmax()) for p in preds]
    if not isinstance(labels[0], int):  # one-hot vectors
        labels = [int(p.to('cpu').argmax()) for p in labels]

    # print(labels)
    # print(preds)
    report = classification_report(labels, preds,
                                   labels=list(range(len(label_list))),
                                   target_names=label_list,
                                   output_dict=True, zero_division=0)
    report = report['macro avg']
    if report_prefix is not None:
        ks = list(report.keys())
        for k in ks:
            report[report_prefix + '_' + k] = report.pop(k)
    return report


# def calculate_metrices(aux_pred_p_total, cls_pred_p_total, labels_total,
#                        hard_exp_pred_total, exp_labels_total, label_list,
#                        num_instances):
#     results = dict()
#     aux_pred_total = [int(p.to('cpu').argmax()) for p in aux_pred_p_total]
#     cls_pred_total = [int(p.to('cpu').argmax()) for p in cls_pred_p_total]
#     labels_total = [int(p.to('cpu').argmax()) for p in labels_total]
#
#     hard_exp_pred_total = np.array([exp.to('cpu').type(torch.int).tolist()
#                                     for exp in hard_exp_pred_total])
#     exp_labels_total = np.array([exp.to('cpu').type(torch.int).tolist()
#                                  for exp in exp_labels_total])
#
#     results['aux_reports'] = classification_report(labels_total, aux_pred_total,
#                                                    labels=list(range(len(label_list))),
#                                                    target_names=label_list,
#                                                    output_dict=True, zero_division=0)
#     results['aux_reports'] = results['aux_reports']['macro avg']
#     results['cls_reports'] = classification_report(labels_total, cls_pred_total,
#                                                    labels=list(range(len(label_list))),
#                                                    target_names=label_list,
#                                                    output_dict=True, zero_division=0)
#     results['cls_reports'] = results['cls_reports']['macro avg']
#     # print(exp_labels_total[0])
#     # print(hard_exp_pred_total[0])
#     exp_p, exp_r, exp_f1 = sequence_macro_prf(exp_labels_total, hard_exp_pred_total)
#
#     results['exp_reports'] = {'precision': exp_p,
#                               'recall': exp_r,
#                               'f1-score': exp_f1,
#                               'support': len(hard_exp_pred_total)}
#
#     # loss, accuracy, optional items
#     num_instances += 1e-9
#     results['loss'] /= num_instances
#     results['aux_loss'] /= num_instances
#     results['cls_loss'] /= num_instances
#     results['exp_loss'] /= num_instances
#     results['aux_acc'] /= num_instances
#     return results
