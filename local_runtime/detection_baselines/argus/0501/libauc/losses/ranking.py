import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import dok_matrix
from .surrogate import squared_loss, squared_hinge_loss, logistic_loss

def _get_surrogate_loss(backend='squared_hinge'):
    if backend == 'squared_hinge':
       surr_loss = squared_hinge_loss
    elif backend == 'squared':
       surr_loss = squared_loss
    elif backend == 'logistic':
       surr_loss = logistic_loss
    else:
        raise ValueError('Out of options!')
    return surr_loss

class ListwiseCE_Loss(torch.nn.Module):
    def __init__(self,
                  id_mapper,
                  total_relevant_pairs,
                  num_pos,
                  gamma0,
                  eps=1e-10,
                  device=None):
        super(ListwiseCE_Loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.id_mapper = id_mapper
        self.num_pos = num_pos
        self.gamma0 = gamma0
        self.eps = eps
        self.u = torch.zeros(total_relevant_pairs).to(self.device)

    def forward(self, predictions, batch):
        batch_size = predictions.size(0)
        neg_pred = torch.repeat_interleave(predictions[:, self.num_pos:], self.num_pos, dim=0)
        pos_pred = torch.cat(torch.chunk(predictions[:, :self.num_pos], batch_size, dim=0), dim=1).permute(1,0)

        margin = neg_pred - pos_pred
        exp_margin = torch.exp(margin - torch.max(margin)).detach_()

        user_ids_tsfd = batch['user_id'].repeat_interleave(self.num_pos)
        pos_item_ids_tsfd = torch.cat(torch.chunk(batch['item_id'][:, :self.num_pos] , batch_size, dim=0), dim=1).squeeze()

        user_item_ids = self.id_mapper[user_ids_tsfd.tolist(), pos_item_ids_tsfd.tolist()].toarray().squeeze()
        self.u[user_item_ids] = (1-self.gamma0) * self.u[user_item_ids] + self.gamma0 * torch.mean(exp_margin, dim=1)

        exp_margin_softmax = exp_margin / (self.u[user_item_ids][:, None] + self.eps)

        loss = torch.sum(margin * exp_margin_softmax)
        loss /= batch_size

        return loss

class NDCG_Loss(torch.nn.Module):
    def __init__(self,
                  id_mapper,
                  total_relevant_pairs,
                  num_user,
                  num_item,
                  num_pos,
                  gamma0,
                  gamma1=0.9,
                  eta0=0.01,
                  margin=1.0,
                  topk=-1,
                  topk_version='theo',
                  tau_1=0.01,
                  tau_2=0.0001,
                  psi_func='sigmoid',
                  topk_margin=2.0,
                  sigmoid_alpha=2.0,
                  surrogate_loss='squared_hinge',
                  device=None):
        super(NDCG_Loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.id_mapper = id_mapper
        self.num_pos = num_pos
        self.margin = margin
        self.gamma0 = gamma0
        self.topk = topk
        self.lambda_q = torch.zeros(num_user+1).to(self.device)
        self.v_q = torch.zeros(num_user+1).to(self.device)
        self.gamma1 = gamma1
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.eta0 = eta0
        self.num_item = num_item
        self.topk_version = topk_version
        self.s_q = torch.zeros(num_user+1).to(self.device)
        self.psi_func = psi_func
        self.topk_margin = topk_margin
        self.sigmoid_alpha = sigmoid_alpha
        self.u = torch.zeros(total_relevant_pairs).to(self.device)
        self.surrogate_loss = _get_surrogate_loss(surrogate_loss)

    def forward(self, predictions, batch):
        device = predictions.device
        ratings = batch['rating'][:, :self.num_pos]
        batch_size = ratings.size()[0]
        predictions_expand = torch.repeat_interleave(predictions, self.num_pos, dim=0)
        predictions_pos = torch.cat(torch.chunk(predictions[:, :self.num_pos], batch_size, dim=0), dim=1).permute(1,0)

        num_pos_items = batch['num_pos_items'].float()
        ideal_dcg = batch['ideal_dcg'].float()

        g = torch.mean(self.surrogate_loss(self.margin, predictions_pos-predictions_expand), dim=-1)
        g = g.reshape(batch_size, self.num_pos)

        G = (2.0 ** ratings - 1).float()

        user_ids = batch['user_id']
        pos_item_ids = batch['item_id'][:, :self.num_pos]

        pos_item_ids = torch.cat(torch.chunk(pos_item_ids, batch_size, dim=0), dim=1).squeeze()
        user_ids_repeat = user_ids.repeat_interleave(self.num_pos)

        user_item_ids = self.id_mapper[user_ids_repeat.tolist(), pos_item_ids.tolist()].toarray().squeeze()
        self.u[user_item_ids] = (1-self.gamma0) * self.u[user_item_ids] + self.gamma0 * g.clone().detach_().reshape(-1)
        g_u = self.u[user_item_ids].reshape(batch_size, self.num_pos)

        nabla_f_g = (G * self.num_item) / ((torch.log2(1 + self.num_item*g_u))**2 * (1 + self.num_item*g_u) * np.log(2))

        if self.topk > 0:
            user_ids = user_ids.long()
            pos_preds_lambda_diffs = predictions[:, :self.num_pos].clone().detach_() - self.lambda_q[user_ids][:, None].to(device)
            preds_lambda_diffs = predictions.clone().detach_() - self.lambda_q[user_ids][:, None].to(device)

            grad_lambda_q = self.topk/self.num_item + self.tau_2*self.lambda_q[user_ids] - torch.mean(torch.sigmoid(preds_lambda_diffs.to(device) / self.tau_1), dim=-1)
            self.v_q[user_ids] = self.gamma1 * grad_lambda_q + (1-self.gamma1) * self.v_q[user_ids]
            self.lambda_q[user_ids] = self.lambda_q[user_ids] - self.eta0 * self.v_q[user_ids]

            if self.topk_version == 'prac':
                nabla_f_g *= torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha)

            elif self.topk_version == 'theo':
                nabla_f_g *= torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha)
                d_psi = torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha) * (1 - torch.sigmoid(pos_preds_lambda_diffs * self.sigmoid_alpha))

                temp_term = torch.sigmoid(preds_lambda_diffs / self.tau_1) * (1 - torch.sigmoid(preds_lambda_diffs / self.tau_1)) / self.tau_1
                L_lambda_hessian = self.tau_2 + torch.mean(temp_term, dim=1)
                self.s_q[user_ids] = self.gamma1 * L_lambda_hessian.to(device) + (1-self.gamma1) * self.s_q[user_ids]
                hessian_term = torch.mean(temp_term * predictions, dim=1) / self.s_q[user_ids].to(device)
                f_g_u = -G / torch.log2(1 + self.num_item*g_u)
                loss = (num_pos_items * torch.mean(nabla_f_g * g + d_psi * f_g_u * (predictions[:, :self.num_pos] - hessian_term[:, None]), dim=-1) / ideal_dcg).mean()
                return loss

        loss = (num_pos_items * torch.mean(nabla_f_g * g, dim=-1) / ideal_dcg).mean()
        return loss

ListwiseCELoss = ListwiseCE_Loss
NDCGLoss = NDCG_Loss
