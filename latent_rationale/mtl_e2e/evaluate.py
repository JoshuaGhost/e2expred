import torch
from collections import defaultdict
from itertools import count
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from latent_rationale.common.util import get_minibatch
from latent_rationale.mtl_e2e.metrics import cls_prf
from latent_rationale.mtl_e2e.models.expred_e2e import HardKumaE2E
from latent_rationale.mtl_e2e.utils import bert_input_preprocess
from latent_rationale.common.util import get_z_stats


def get_histogram_counts(z=None, mask=None, mb=None):
    counts = np.zeros(2).astype(np.int64)

    for i, ex in enumerate(mb):

        tokens = ex.tokens
        token_labels = ex.token_labels

        if z is not None:
            ex_z = z[i][:len(tokens)]

        if mask is not None:
            assert mask[i].sum() == len(tokens), "mismatch mask/tokens"

        for j, tok, lab in zip(count(), tokens, token_labels):
            if z is not None:
                if ex_z[j] > 0:
                    counts[lab] += 1
            else:
                counts[lab] += 1

    return counts


# metrics_names = "aux_acc cls_acc cls_f1 exp_acc exp_precision exp_recall exp_f1 best_eval".split()


def evaluate(model:HardKumaE2E, data, tokenizer,
             weights, label_list,
             batch_size=25,
             max_length=512,
             device=None,
             tolerance=0) -> defaultdict:
    """Accuracy of a model on given data set (using minibatches)"""

    model.eval()  # disable dropout

    # z statistics
    results = defaultdict(float)
    z_totals = defaultdict(float)
    # histogram_totals = np.zeros(5).astype(np.int64)
    # z_histogram_totals = np.zeros(5).astype(np.int64)

    aux_pred_p_total = []
    cls_pred_p_total = []
    soft_exp_pred_total = []
    hard_exp_pred_total = []
    masks_total = []
    labels_total = []
    exp_labels_total = []
    for batch in data:
        inputs, exp_labels, cls_labels, positions, attention_masks, query_masks = batch

        exp_labels = exp_labels.cuda()
        cls_labels = cls_labels.cuda()

        # inputs, exp_labels, cls_labels, positions, attention_masks, padding_masks = bert_input_preprocess(batch,
        #                                                                                           tokenizer=tokenizer,
        #                                                                                           max_length=max_length,
        #                                                                                           device=device)
        batch_size = len(batch)
        # print(inputs.data.shape)
        with torch.no_grad():
            aux_pred_p, cls_pred_p, soft_exp_pred, hard_exp_pred = model(inputs.data,
                                                                         attention_masks=attention_masks,
                                                                         positions=positions.data,
                                                                         query_masks=query_masks)
            aux_pred_p_total.extend(aux_pred_p)
            cls_pred_p_total.extend(cls_pred_p)
            soft_exp_pred_total.extend(soft_exp_pred)
            hard_exp_pred_total.extend(hard_exp_pred)
            masks_total.extend(attention_masks)
            labels_total.extend(cls_labels)
            exp_labels_total.extend(exp_labels.data)
            loss, loss_optional = model.get_loss(aux_pred_p, cls_pred_p, cls_labels,
                                                 hard_exp_pred, exp_labels.data,
                                                 mask=attention_masks,
                                                 weights=weights,
                                                 tolerance=tolerance)

            results['loss'] += loss.item() * batch_size
            for k, v in loss_optional.items():
                if not isinstance(v, float):
                    v = v.item()
                results[k] += v * batch_size

            if hasattr(model, "z"):
                n0, nc, n1, nt = get_z_stats(soft_exp_pred, attention_masks)
                z_totals['p0'] += n0
                z_totals['pc'] += nc
                z_totals['p1'] += n1
                z_totals['total'] += nt
                # histogram counts
                # for this need to sort z in original order
                # z = model.z.squeeze(1).squeeze(-1)
                # z_histogram = get_histogram_counts(z=z, mask=attention_masks, mb=batch)
                # z_histogram_totals += z_histogram
                # histogram = get_histogram_counts(mb=batch)
                # histogram_totals += histogram
        results['total'] += batch_size

    aux_pred_total = [int(p.to('cpu').argmax(axis=-1)) for p in aux_pred_p_total]
    cls_pred_total = [int(p.to('cpu').argmax(axis=-1)) for p in cls_pred_p_total]
    # print(labels_total)
    labels_total = [int(p.to('cpu')) for p in labels_total]
    # print(labels_total)

    eps = 1e-10
    total = results['total'] + eps
    for k, v in results.items():
        if k != "total":
            results[k] = v / total

    results.update(cls_prf(labels_total, aux_pred_total, label_list, report_prefix='aux'))
    results.update(cls_prf(labels_total, cls_pred_total, label_list, report_prefix='cls'))

    # loss, accuracy, optional items
    z_totals['total'] += 1e-9
    for k, v in z_totals.items():
        if k != "total":
            results[k] = v / z_totals["total"]

    if "p0" in results:
        results["selected"] = 1 - results["p0"]

    return results
