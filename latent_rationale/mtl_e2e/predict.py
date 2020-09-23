import numpy as np
import torch

from latent_rationale.common.util import get_minibatch
from latent_rationale.mtl_e2e.utils import prepare_minibatch


def predict(model, data, tokenizer, batch_size=25, max_length=512, device=None):
    """Accuracy of a model on given data set (using minibatches)"""

    model.eval()  # disable dropout

    # z statistics
    # histogram_totals = np.zeros(5).astype(np.int64)
    # z_histogram_totals = np.zeros(5).astype(np.int64)

    # aux_pred_total = []
    cls_pred_p_total = []
    hard_exp_pred_total = []
    soft_exp_pred_total = []
    # masks_total = []
    # labels_total = []
    # exp_labels_total = []
    for batch in get_minibatch(data, batch_size=batch_size, shuffle=False):
        inputs, exp_labels, labels, positions, attention_masks, padding_masks = prepare_minibatch(batch,
                                                                                                  tokenizer=tokenizer,
                                                                                                  max_length=max_length,
                                                                                                  device=device)
        with torch.no_grad():
            aux_pred_p, cls_pred_p, soft_exp_pred, hard_exp_pred = model(inputs.data,
                                                                         attention_masks=attention_masks,
                                                                         padding_masks=padding_masks,
                                                                         positions=positions.data)
            # aux_pred_total.extend(aux_pred)
            cls_pred_p_total.extend(cls_pred_p)
            hard_exp_pred_total.extend(hard_exp_pred)
            soft_exp_pred_total.extend(soft_exp_pred)
            # masks_total.extend(attention_masks)
            # labels_total.extend(labels)
            # exp_labels.extend(exp_labels)
    return cls_pred_p_total, soft_exp_pred_total, hard_exp_pred_total
