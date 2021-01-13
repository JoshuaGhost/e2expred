from typing import Any

from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
import torch
from torch import nn
from latent_rationale.common.util import get_z_stats

from latent_rationale.mtl_e2e.models.components import BertClassifier, KumaSelectorLayer


class HardKumaE2E(nn.Module):
    def __init__(self,
                 cfg: Any,
                 tokenizer: BertTokenizer,
                 epochs_total: int):
        super(HardKumaE2E, self).__init__()

        self.epochs_total = epochs_total
        self.weights_scheduler = cfg['weights_scheduler']
        self.lambda_min = cfg['lambda_min']
        self.lambda_max = cfg['lambda_max']
        mtl_params = cfg['mtl']
        self.bert_mtl = BertClassifier(mtl_params, tokenizer)
        # self.bert_mtl_exp = nn.Linear(300, 768)
        self.auxilliary_criterion = nn.CrossEntropyLoss(reduction='none')

        bert_ebd_dims = self.bert_mtl.bert_model.config.hidden_size
        selector_params = cfg['selector']
        selector_type = cfg['selector_type']
        if selector_type == 'hard binary':
            self.rationale_selector = KumaSelectorLayer(selector_params,
                                                        repr_size=bert_ebd_dims)
        else:
            raise NotImplementedError
        from latent_rationale.common.losses import resampling_rebalanced_crossentropy
        self.exp_criterion = resampling_rebalanced_crossentropy(seq_reduction='mean')
        self.exp_threshold = cfg.get('exp_threshold', 0.5)

        classifier_params = cfg['classifier']
        self.shared_encoder = cfg.get('share_encoder', False)
        if self.shared_encoder:
            self.classifier = self.bert_mtl
        else:
            self.classifier = BertClassifier(classifier_params, tokenizer)
        self.final_cls_criterion = nn.CrossEntropyLoss(reduction='none')

        self.soft_selection = cfg['soft_selection']

        if cfg.get('warm_start_mtl', None) is not None:
            self.bert_mtl.bert_model.load_state_dict(torch.load(cfg['warm_start_mtl']))
        if cfg.get('warm_start_cls', None) is not None and not self.shared_encoder:
            # If the encoder is shared, the cls warm start is muted
            self.classifier.bert_model.load_state_dict(torch.load(cfg['warm_start_cls']))

        lambda_init = cfg['lambda_init']
        self.register_buffer('lambda0', torch.full((1,), lambda_init))
        self.register_buffer('lambda1', torch.full((1,), lambda_init))
        self.register_buffer('c0_ma', torch.full((1,), 0.))  # moving average
        self.register_buffer('c1_ma', torch.full((1,), 0.))

    @classmethod
    def new(cls, cfg, tokenizer, epochs_total):
        return cls(cfg, tokenizer, epochs_total)

    def forward(self,
                bert_inputs: torch.Tensor,
                attention_masks: torch.Tensor,
                padding_masks: torch.Tensor,
                positions: torch.Tensor):
        device = next(self.parameters()).device

        auxiliary_cls_p, mtl_exp_output_hidden = self.bert_mtl(bert_inputs,
                                                               attention_masks=attention_masks,
                                                               positions=positions)
        soft_z = self.rationale_selector(mtl_exp_output_hidden, padding_masks).squeeze(dim=-1)
        if self.training:
            hard_z = torch.where(soft_z >= self.exp_threshold, torch.LongTensor([1]).to(device=device),
                                 torch.LongTensor([0]).to(device=device))
            hard_z = (hard_z - soft_z).detach() + soft_z
        else:
            hard_z = torch.where(soft_z >= self.exp_threshold, torch.LongTensor([1]).to(device=device),
                                 torch.LongTensor([0]).to(device=device))
        if self.soft_selection:
            attribution_map = soft_z
        else:
            attribution_map = hard_z
        cls_p, _ = self.classifier(bert_inputs,
                                   attention_masks=attention_masks,
                                   embedding_masks=attribution_map,
                                   positions=positions)
        return auxiliary_cls_p, cls_p, soft_z, hard_z

    def get_loss(self, p_aux, p_cls, t_cls, p_exp, t_exp, mask, weights, epoch=None):
        optional = {}
        selection = weights['selection']
        lasso = weights['lasso']
        lagrange_alpha = weights['lagrange_alpha']
        lagrange_lr = weights['lagrange_lr']
        batch_size = mask.size(0)
        lengths = mask.sum(1).float()  # [B]

        w_aux = weights.get('w_aux', 1.)
        w_exp = weights.get('w_exp', 1.)
        w_cls = 1.

        if self.training:
            if self.weights_scheduler == 'parabolic':
                # This weights scheduler is inspired by the paper
                # https://arxiv.org/abs/1912.02413
                decay_classifier = (float(epoch - 1) / self.epochs_total) ** 2
                decay_identifier = 1 - decay_classifier
                w_aux *= decay_identifier
                w_exp *= decay_identifier
                w_cls *= decay_classifier
            elif self.scheduling == 'static':
                w_aux = weights['aux']
                w_exp = weights['exp']
                w_cls = weights['cls']

        # print(f"w_aux: {w_aux}, w_exp: {w_exp}, w_cls: {w_cls}")
        # print(self.training, self.epochs_total, epoch)

        loss_aux = self.auxilliary_criterion(p_aux, t_cls).mean()
        optional['aux_acc'] = accuracy_score(t_cls.cpu(), p_aux.cpu().argmax(axis=-1).detach())
        optional["aux_loss"] = loss_aux.item()
        loss = w_aux * loss_aux

        loss_exp = self.exp_criterion(p_exp, t_exp).mean()
        optional["exp_loss"] = loss_exp.item()
        eps = 1e-10
        optional['exp_p'] = (((p_exp > self.exp_threshold).type(torch.long) * t_exp).sum(-1)/(p_exp.sum(-1)+eps)).mean()
        optional['exp_r'] = (((p_exp > self.exp_threshold).type(torch.long) * t_exp).sum(-1)/(t_exp.sum(-1)+eps)).mean()
        loss += w_exp * loss_exp

        loss_cls = self.final_cls_criterion(p_cls, t_cls).mean()
        optional['cls_acc'] = accuracy_score(t_cls.cpu(), p_cls.cpu().argmax(axis=-1).detach())
        optional["cls_loss"] = loss_cls.item()
        loss += w_cls * loss_cls

        z_dists = self.rationale_selector.z_dists

        # pre-compute for regularizers: pdf(0.)
        if len(z_dists) == 1:
            pdf0 = z_dists[0].pdf(0.)
        else:
            pdf0 = []
            for t in range(len(z_dists)):
                pdf_t = z_dists[t].pdf(0.)
                pdf0.append(pdf_t)
            pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]

        pdf0 = pdf0.squeeze(-1)
        pdf0 = torch.where(mask, pdf0, pdf0.new_zeros([1]))  # [B, T]

        # L0 regularizer
        pdf_nonzero = 1. - pdf0  # [B, T]
        pdf_nonzero = torch.where(mask, pdf_nonzero, pdf_nonzero.new_zeros([1]))

        if self.lambda0 > 0.:
            l0 = pdf_nonzero.sum(1) / (lengths + 1e-9)  # [B]
            l0 = l0.sum() / batch_size

            # `l0` now has the expected selection rate for this mini-batch
            # we now follow the steps Algorithm 1 (page 7) of this paper:
            # https://arxiv.org/abs/1810.00597
            # to enforce the constraint that we want l0 to be not higher
            # than `selection` (the target sparsity rate)

            # lagrange dissatisfaction, batch average of the constraint
            c0_hat = (l0 - selection)

            # moving average of the constraint
            self.c0_ma = lagrange_alpha * self.c0_ma + \
                         (1 - lagrange_alpha) * c0_hat.item()

            # compute smoothed constraint (equals moving average c0_ma)
            c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())

            # update lambda
            self.lambda0 = self.lambda0 * torch.exp(
                lagrange_lr * c0.detach())
            self.lambda0 = self.lambda0.clamp(self.lambda_min, self.lambda_max)

            with torch.no_grad():
                optional["cost0_l0"] = l0.item()
                optional["target_selection"] = selection
                optional["c0_hat"] = c0_hat.item()
                optional["c0"] = c0.item()  # same as moving average
                optional["lambda0"] = self.lambda0.item()
                optional["lagrangian0"] = (self.lambda0 * c0_hat).item()
                optional["a"] = z_dists[0].a.mean().item()
                optional["b"] = z_dists[0].b.mean().item()

            loss = loss + self.lambda0.detach() * c0

        if lasso > 0.:
            # fused lasso (coherence constraint)

            # cost z_t = 0, z_{t+1} = non-zero
            zt_zero = pdf0[:, :-1]
            ztp1_nonzero = pdf_nonzero[:, 1:]

            # cost z_t = non-zero, z_{t+1} = zero
            zt_nonzero = pdf_nonzero[:, :-1]
            ztp1_zero = pdf0[:, 1:]

            # number of transitions per sentence normalized by length
            lasso_cost = zt_zero * ztp1_nonzero + zt_nonzero * ztp1_zero
            lasso_cost = lasso_cost * mask.float()[:, :-1]
            lasso_cost = lasso_cost.sum(1) / (lengths + 1e-9)  # [B]
            lasso_cost = lasso_cost.sum() / batch_size

            # lagrange coherence dissatisfaction (batch average)
            target1 = lasso

            # lagrange dissatisfaction, batch average of the constraint
            c1_hat = (lasso_cost - target1)

            # update moving average
            self.c1_ma = lagrange_alpha * self.c1_ma + \
                         (1 - lagrange_alpha) * c1_hat.detach()

            # compute smoothed constraint
            c1 = c1_hat + (self.c1_ma.detach() - c1_hat.detach())

            # update lambda
            self.lambda1 = self.lambda1 * torch.exp(
                lagrange_lr * c1.detach())

            with torch.no_grad():
                optional["cost1_lasso"] = lasso_cost.item()
                optional["lasso"] = lasso
                optional["c1_hat"] = c1_hat.item()
                optional["c1"] = c1.item()  # same as moving average
                optional["lambda1"] = self.lambda1.item()
                optional["lagrangian1"] = (self.lambda1 * c1_hat).item()

            loss = loss + self.lambda1.detach() * c1

        # z statistics
        if self.training:
            num_0, num_c, num_1, total = get_z_stats(self.rationale_selector.z, mask)
            optional["p0"] = num_0 / float(total)
            optional["pc"] = num_c / float(total)
            optional["p1"] = num_1 / float(total)
            optional["selected"] = 1 - optional["p0"]

        return loss, optional
