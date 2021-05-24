from typing import Tuple, Dict

from sklearn.metrics import accuracy_score
from transformers import BertTokenizer
import torch
from torch import nn
from latent_rationale.common.util import get_z_stats
from latent_rationale.mtl_e2e.config import E2ExPredConfig

from latent_rationale.mtl_e2e.models.components import BertClassifier, KumaSelectorLayer


class HardKumaE2E(nn.Module):
    def __init__(self,
                 cfg: E2ExPredConfig,
                 tokenizer: BertTokenizer,
                 epochs_total: int):
        super(HardKumaE2E, self).__init__()

        self.epochs_total = epochs_total
        self.weights_scheduler = cfg.weights_scheduler
        self.lambda_min = cfg.lambda_min
        self.lambda_max = cfg.lambda_max
        mtl_params = cfg.mtl_conf
        self.bert_mtl = BertClassifier(mtl_params, tokenizer)
        # self.bert_mtl_exp = nn.Linear(300, 768)
        self.auxilliary_criterion = nn.CrossEntropyLoss(reduction='none')

        bert_ebd_dims = self.bert_mtl.bert_model.config.hidden_size
        selector_params = cfg.selector_conf
        if selector_params.selector_type == 'hard binary':
            self.rationale_selector = KumaSelectorLayer(selector_params,
                                                        repr_size=bert_ebd_dims)
        else:
            raise NotImplementedError
        from latent_rationale.common.losses import resampling_rebalanced_crossentropy
        self.exp_criterion = resampling_rebalanced_crossentropy(seq_reduction='mean')
        self.exp_threshold = selector_params.exp_threshold

        classifier_params = cfg.cls_conf
        self.share_encoder = cfg.share_encoder
        if self.share_encoder:
            self.classifier = self.bert_mtl
        else:
            self.classifier = BertClassifier(classifier_params, tokenizer)
        self.final_cls_criterion = nn.CrossEntropyLoss(reduction='none')

        self.soft_selection = selector_params.soft_selection

        if cfg.mtl_conf.warm_start:
            state_dict = torch.load(cfg.mtl_conf.pretrained_model_dir)
            self.bert_mtl.bert_model.load_state_dict(state_dict)
        if cfg.cls_conf.warm_start and not self.share_encoder:
            # If the encoder is shared, the cls warm start is muted
            state_dict = torch.load(cfg.cls_conf.pretrained_model_dir)
            self.classifier.bert_model.load_state_dict(state_dict)

        lambda_init = cfg.lambda_init
        self.register_buffer('lambda_exp', torch.full((1,), lambda_init))
        self.register_buffer('lambda1', torch.full((1,), lambda_init))
        self.register_buffer('loss_exp_ma', torch.full((1,), 0.))  # moving average of exp_loss
        self.register_buffer('c1_ma', torch.full((1,), 0.))

    @classmethod
    def new(cls, cfg, tokenizer, epochs_total):
        return cls(cfg, tokenizer, epochs_total)

    def forward(self,
                inputs: torch.Tensor,
                attention_masks: torch.Tensor,
                positions: torch.Tensor,
                query_masks: torch.Tensor=None):
        device = next(self.parameters()).device

        inputs = inputs.to(device)
        attention_masks = attention_masks.to(device)
        positions = positions.to(device)

        auxiliary_cls_p, mtl_exp_output_hidden = self.bert_mtl(inputs,
                                                               attention_masks=attention_masks,
                                                               positions=positions)
        soft_z = self.rationale_selector(mtl_exp_output_hidden, attention_masks).squeeze(dim=-1)
        hard_z = torch.where(soft_z >= self.exp_threshold,
                             torch.ones_like(soft_z).to(device=device),
                             torch.zeros_like(soft_z).to(device=device))
        if query_masks is not None:
            hard_z = torch.where(query_masks.to(device=device),
                                 torch.ones_like(soft_z).to(device=device),
                                 hard_z)
        if self.training:
            hard_z = (hard_z - soft_z).detach() + soft_z
        # else:
        #     hard_z = torch.where(soft_z >= self.exp_threshold, torch.LongTensor([1]).to(device=device),
        #                          torch.LongTensor([0]).to(device=device))
        if self.soft_selection:
            attribution_map = soft_z
        else:
            attribution_map = hard_z
        cls_p, _ = self.classifier(inputs,
                                   attention_masks=attention_masks,
                                   embedding_masks=attribution_map,
                                   positions=positions)
        return auxiliary_cls_p, cls_p, soft_z, hard_z

    def get_loss(
            self, p_aux, p_cls, t_cls, p_exp, t_exp, mask: torch.Tensor, weights, epoch=None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        optional = {}
        # selection = weights['selection']
        # lasso = weights['lasso']
        lagrange_alpha = weights['lagrange_alpha']
        exp_lagrange_lr = weights['exp_lagrange_lr']
        # batch_size = mask.size(0)
        # lengths = mask.sum(1).float()  # [B]

        w_aux = weights.get('w_aux', 1.)
        w_exp = weights.get('w_exp', 1.)
        w_cls = weights.get('w_cls', 1.)

        if self.training and self.weights_scheduler == 'parabolic':
            # This weights scheduler is inspired by the paper
            # https://arxiv.org/abs/1912.02413
            decay_classifier = (float(epoch - 1) / self.epochs_total) ** 2
            decay_identifier = 1 - decay_classifier
            # w_aux *= decay_identifier
            # w_exp *= decay_identifier
            # w_cls *= decay_classifier
        else:
            decay_identifier, decay_classifier = 1., 1.
            # w_aux = weights['aux']
            # w_exp = weights['exp']
            # w_cls = weights['cls']

        loss_aux = self.auxilliary_criterion(p_aux, t_cls).mean()
        optional['aux_acc'] = accuracy_score(t_cls.cpu(), p_aux.cpu().argmax(axis=-1).detach())
        optional["aux_loss"] = loss_aux.item()
        loss = w_aux * decay_identifier * loss_aux

        loss_exp = self.exp_criterion(p_exp, t_exp).mean()
        optional["exp_loss"] = loss_exp.item()
        eps = 1e-10
        optional['exp_p'] = (
                    ((p_exp > self.exp_threshold).type(torch.long) * t_exp).sum(-1) / (p_exp.sum(-1) + eps)).mean()
        optional['exp_r'] = (
                    ((p_exp > self.exp_threshold).type(torch.long) * t_exp).sum(-1) / (t_exp.sum(-1) + eps)).mean()
        optional['exp_f1'] = 2. / (1. / (optional['exp_p'] + eps) + 1. / (optional['exp_r'] + eps))
        if self.lambda_exp > 0.:
            # The exp loss is now considered as a Lagrangian constrain.
            # we now follow the steps Algorithm 1 (page 7) of this paper:
            # https://arxiv.org/abs/1810.00597
            # to enforce the exp constraint as low as possible

            # moving average of the constraint
            self.loss_exp_ma = lagrange_alpha * self.loss_exp_ma + \
                               (1 - lagrange_alpha) * loss_exp.item()

            # compute smoothed constraint (equals moving average loss_exp_ma)
            loss_exp_reg = loss_exp + (self.loss_exp_ma.item() - loss_exp.item())

            # update lambda_exp
            self.lambda_exp = self.lambda_exp * torch.exp(
                exp_lagrange_lr * loss_exp_reg.detach())
            self.lambda_exp = self.lambda_exp.clamp(self.lambda_min, self.lambda_max)

            with torch.no_grad():
                optional["loss_exp_reg"] = loss_exp_reg.item()  # same as moving average
                optional["lambda_exp"] = self.lambda_exp.item()
                optional["lagrangian_exp"] = (self.lambda_exp * loss_exp).item()
            loss += self.lambda_exp.item() * decay_identifier * loss_exp_reg

        loss_cls = self.final_cls_criterion(p_cls, t_cls).mean()
        optional['cls_acc'] = accuracy_score(t_cls.cpu(), p_cls.cpu().argmax(axis=-1).detach())
        optional["cls_loss"] = loss_cls.item()
        loss += w_cls * decay_classifier * loss_cls

        # z_dists = self.rationale_selector.z_dists

        # pre-compute for regularizers: pdf(0.)
        # if len(z_dists) == 1:
        #     pdf0 = z_dists[0].pdf(0.)
        # else:
        #     pdf0 = []
        #     for t in range(len(z_dists)):
        #         pdf_t = z_dists[t].pdf(0.)
        #         pdf0.append(pdf_t)
        #     pdf0 = torch.stack(pdf0, dim=1)  # [B, T, 1]
        #
        # pdf0 = pdf0.squeeze(-1)
        # pdf0 = torch.where(mask, pdf0, pdf0.new_zeros([1]))  # [B, T]
        #
        # # L0 regularizer
        # pdf_nonzero = 1. - pdf0  # [B, T]
        # pdf_nonzero = torch.where(mask, pdf_nonzero, pdf_nonzero.new_zeros([1]))

        # if self.lambda0 > 0.:
        # if False:
        #     l0 = pdf_nonzero.sum(1) / (lengths + 1e-9)  # [B]
        #     l0 = l0.sum() / batch_size
        #
        #     # `l0` now has the expected selection rate for this mini-batch
        #     # we now follow the steps Algorithm 1 (page 7) of this paper:
        #     # https://arxiv.org/abs/1810.00597
        #     # to enforce the constraint that we want l0 to be not higher
        #     # than `selection` (the target sparsity rate)
        #
        #     # lagrange dissatisfaction, batch average of the constraint
        #     c0_hat = (l0 - selection)
        #
        #     # moving average of the constraint
        #     self.c0_ma = lagrange_alpha * self.c0_ma + \
        #                  (1 - lagrange_alpha) * c0_hat.item()
        #
        #     # compute smoothed constraint (equals moving average c0_ma)
        #     c0 = c0_hat + (self.c0_ma.detach() - c0_hat.detach())
        #
        #     # update lambda
        #     self.lambda0 = self.lambda0 * torch.exp(
        #         l0_lagrange_lr * c0.detach())
        #     self.lambda0 = self.lambda0.clamp(self.lambda_min, self.lambda_max)
        #
        #     with torch.no_grad():
        #         optional["cost0_l0"] = l0.item()
        #         optional["target_selection"] = selection
        #         optional["c0_hat"] = c0_hat.item()
        #         optional["c0"] = c0.item()  # same as moving average
        #         optional["lambda0"] = self.lambda0.item()
        #         optional["lagrangian0"] = (self.lambda0 * c0_hat).item()
        #         optional["a"] = z_dists[0].a.mean().item()
        #         optional["b"] = z_dists[0].b.mean().item()
        #
        #     loss = loss + self.lambda0.detach() * c0
        #
        # if lasso > 0.:
        #     # fused lasso (coherence constraint)
        #
        #     # cost z_t = 0, z_{t+1} = non-zero
        #     zt_zero = pdf0[:, :-1]
        #     ztp1_nonzero = pdf_nonzero[:, 1:]
        #
        #     # cost z_t = non-zero, z_{t+1} = zero
        #     zt_nonzero = pdf_nonzero[:, :-1]
        #     ztp1_zero = pdf0[:, 1:]
        #
        #     # number of transitions per sentence normalized by length
        #     lasso_cost = zt_zero * ztp1_nonzero + zt_nonzero * ztp1_zero
        #     lasso_cost = lasso_cost * mask.float()[:, :-1]
        #     lasso_cost = lasso_cost.sum(1) / (lengths + 1e-9)  # [B]
        #     lasso_cost = lasso_cost.sum() / batch_size
        #
        #     # lagrange coherence dissatisfaction (batch average)
        #     target1 = lasso
        #
        #     # lagrange dissatisfaction, batch average of the constraint
        #     c1_hat = (lasso_cost - target1)
        #
        #     # update moving average
        #     self.c1_ma = lagrange_alpha * self.c1_ma + \
        #                  (1 - lagrange_alpha) * c1_hat.detach()
        #
        #     # compute smoothed constraint
        #     c1 = c1_hat + (self.c1_ma.detach() - c1_hat.detach())
        #
        #     # update lambda
        #     self.lambda1 = self.lambda1 * torch.exp(
        #         l0_lagrange_lr * c1.detach())
        #
        #     with torch.no_grad():
        #         optional["cost1_lasso"] = lasso_cost.item()
        #         optional["lasso"] = lasso
        #         optional["c1_hat"] = c1_hat.item()
        #         optional["c1"] = c1.item()  # same as moving average
        #         optional["lambda1"] = self.lambda1.item()
        #         optional["lagrangian1"] = (self.lambda1 * c1_hat).item()
        #
        #     loss = loss + self.lambda1.detach() * c1

        # z statistics
        if self.training:
            num_0, num_c, num_1, total = get_z_stats(self.rationale_selector.z, mask)
            optional["p0"] = num_0 / float(total)
            optional["pc"] = num_c / float(total)
            optional["p1"] = num_1 / float(total)
            optional["selected"] = 1 - optional["p0"]

        return loss, optional
