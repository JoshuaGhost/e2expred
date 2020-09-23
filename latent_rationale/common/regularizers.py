import torch


def l0_for_hardkuma(dists, mask):
    batch_size = mask.size(0)
    lengths = mask.sum(1).float()
    # pre-compute for regularizers: pdf(0.)
    if len(dists) == 1:
        pdf0 = dists[0].pdf(0.)
    else:
        pdf0 = []
        for t in range(len(dists)):
            pdf_t = dists[t].pdf(0.)
            pdf0.append(pdf_t)
        pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]

    pdf0 = pdf0.squeeze(-1)
    pdf0 = torch.where(mask, pdf0, pdf0.new_zeros([1]))  # [B, T]

    # L0 regularizer
    pdf_nonzero = 1. - pdf0  # [B, T]
    pdf_nonzero = torch.where(mask, pdf_nonzero, pdf_nonzero.new_zeros([1]))

    l0 = pdf_nonzero.sum(1) / (lengths + 1e-9)  # [B]
    l0 = l0.sum() / batch_size
    return l0