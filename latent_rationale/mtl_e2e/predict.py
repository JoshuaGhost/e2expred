import numpy as np
import torch

from latent_rationale.common.util import get_minibatch
from latent_rationale.mtl_e2e.utils import bert_input_preprocess


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
    for batch in data:
        inputs, _, _, positions, attention_masks, query_masks = batch

        # exp_labels = exp_labels.cuda()
        # cls_labels = cls_labels.cuda()
        # inputs, exp_labels, cls_labels, positions, attention_masks, padding_masks = bert_input_preprocess(batch,
        #                                                                                           tokenizer=tokenizer,
        #                                                                                           max_length=max_length,
        #                                                                                           device=device)
        with torch.no_grad():
            aux_pred_p, cls_pred_p, soft_exp_pred, hard_exp_pred = model(inputs=inputs,
                                                                         attention_masks=attention_masks,
                                                                         positions=positions.data,
                                                                         query_masks=query_masks)
            # aux_pred_total.extend(aux_pred)
            cls_pred_p_total.extend(cls_pred_p)
            hard_exp_pred_total.extend(hard_exp_pred)
            soft_exp_pred_total.extend(soft_exp_pred)
            # masks_total.extend(attention_masks)
            # labels_total.extend(labels)
            # exp_labels.extend(exp_labels)
    return cls_pred_p_total, soft_exp_pred_total, hard_exp_pred_total
