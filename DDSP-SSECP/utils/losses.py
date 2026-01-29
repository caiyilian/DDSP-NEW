import torch

from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math


def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def w_gradient_loss(s, w, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    wy = w[:, :, 1:, :, :]
    wx = w[:, :, :, 1:, :]
    wz = w[:, :, :, :, 1:]

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.sum(wx * dx) / (torch.sum(wx) + 1) + torch.sum(wy * dy) / (torch.sum(wy) + 1) + torch.sum(wz * dz) / (
                torch.sum(wz) + 1)
    return d / 3.0


def dice_coef(y_true, y_pred, mask=None):
    smooth = 1.

    if mask is not None:
        a = torch.sum(y_true * y_pred * mask, (2, 3, 4))
        b = torch.sum(y_true ** 2 * mask, (2, 3, 4))
        c = torch.sum(y_pred ** 2 * mask, (2, 3, 4))
    else:
        a = torch.sum(y_true * y_pred, (2, 3, 4))
        b = torch.sum(y_true ** 2, (2, 3, 4))
        c = torch.sum(y_pred ** 2, (2, 3, 4))

    dice = (2 * a + smooth) / (b + c + smooth)

    return torch.mean(dice)


def dice_loss(y_true, y_pred, mask=None):
    d = dice_coef(y_true, y_pred, mask)
    return 1 - d


def MSE(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def mix_ce_dice(y_true, y_pred):
    return crossentropy(y_true, y_pred) + 1 - dice_coef(y_true, y_pred)


def prob_entropyloss(pred):
    pred = pred + 1e-5
    out = - pred * torch.log(pred)
    return torch.mean(out)

def prob_entropyloss_sel(pred):
    pred = pred + 1e-5
    out = - pred * torch.log(pred)
    return out


def crossentropy(y_pred, y_true):
    smooth = 1e-6
    return -torch.mean(y_true * torch.log(y_pred + smooth))


def B_crossentropy(y_pred, y_true):
    smooth = 1e-6
    return -torch.mean(y_true * torch.log(y_pred + smooth) + (1 - y_true) * torch.log(1 - y_pred + smooth))


def select_ent_loss(score_t_og_ori, aug_preds, batch_img_shape, q_len):
    score_t_og = score_t_og_ori.detach()
    tgt_preds = score_t_og.max(dim=1)[1]
    correct_mask, incorrect_mask = torch.zeros_like(tgt_preds), torch.zeros_like(tgt_preds)
    score_t_aug_pos, score_t_aug_neg = torch.zeros_like(score_t_og), torch.zeros_like(score_t_og)  ###B*C*H*W

    for aug_pred in aug_preds:
        # 投票
        tgt_preds_aug = aug_pred.max(dim=1)[1]
        consistent_idxs = (tgt_preds == tgt_preds_aug).detach()
        inconsistent_idxs = (tgt_preds != tgt_preds_aug).detach()
        correct_mask = correct_mask + consistent_idxs.type(torch.uint8)
        incorrect_mask = incorrect_mask + inconsistent_idxs.type(torch.uint8)

        # 下边的repeat可能维度(1, 5, 1, 1)有问题
        # 雀食，因为是体素，所以要多加1维；这里的c即为预测的类别总数（应该是吧）
        ############################################################################
        # 框中这部分还是有问题，得看源代码理解下
        # 下两行操作我懂，通道加1维，并复制
        # consistent_idxs_x = consistent_idxs.unsqueeze(1).repeat(1, 5, 1, 1, 1)  ###B*C*H*W
        # inconsistent_idxs_x = inconsistent_idxs.unsqueeze(1).repeat(1, 5, 1, 1, 1)  ###B*C*H*W
        #
        # score_t_aug_pos = torch.where(consistent_idxs_x, aug_pred, score_t_aug_pos)  ###B*C*H*W
        # score_t_aug_neg = torch.where(inconsistent_idxs_x, aug_pred, score_t_aug_neg)  ####B*C*H*W
        ############################################################################
    correct_mask, incorrect_mask = correct_mask > (q_len // 2), incorrect_mask > (q_len // 2)

    # B*H*W得到像素总数目，correct_mask全部求和得到预测正确的数目
    correct_ratio = (correct_mask).sum().item() / \
                    (batch_img_shape[0] * batch_img_shape[2] * batch_img_shape[3] * batch_img_shape[4])
    incorrect_ratio = (incorrect_mask).sum().item() / \
                      (batch_img_shape[0] * batch_img_shape[2] * batch_img_shape[3] * batch_img_shape[4])

    probs_t_sum = prob_entropyloss_sel(score_t_og_ori)
    if correct_ratio > 0.0:
        # probs_t_pos = F.softmax(score_t_aug_pos, dim=1)

        # print(torch.masked_select(probs_t_pos_sum, correct_mask).shape)
        loss_cent_correct = 1 * correct_ratio * torch.mean(torch.masked_select(probs_t_sum, correct_mask))
    else:
        loss_cent_correct = 0

    if incorrect_ratio > 0.0:
        # probs_t_neg = F.softmax(score_t_aug_neg, dim=1)
        # probs_t_neg_sum = prob_entropyloss_sel(score_t_og_ori)
        loss_cent_incorrect = 1 * incorrect_ratio * -torch.mean(torch.masked_select(probs_t_sum, incorrect_mask))
        # print(torch.masked_select(probs_t_neg_sum, incorrect_mask).shape)
    else:
        loss_cent_incorrect = 0

    return loss_cent_correct + loss_cent_incorrect


def select_ent_loss2(score_t_og_ori, aug_preds, batch_img_shape, q_len):  # 这里的计算不对，得改
    score_t_og = score_t_og_ori.detach()
    tgt_preds = score_t_og.max(dim=1)[1]
    correct_mask, incorrect_mask = torch.zeros_like(tgt_preds), torch.zeros_like(tgt_preds)
    score_t_aug_pos, score_t_aug_neg = torch.zeros_like(score_t_og), torch.zeros_like(score_t_og)  ###B*C*H*W

    for aug_pred in aug_preds:
        # 投票（这里没问题，而且这里仿照源代码写的还稍显冗余）
        tgt_preds_aug = aug_pred.max(dim=1)[1]
        consistent_idxs = (tgt_preds == tgt_preds_aug).detach()
        inconsistent_idxs = (tgt_preds != tgt_preds_aug).detach()
        correct_mask = correct_mask + consistent_idxs.type(torch.uint8)
        incorrect_mask = incorrect_mask + inconsistent_idxs.type(torch.uint8)

        # 下边的repeat可能维度(1, 5, 1, 1)有问题
        # 雀食，因为是体素，所以要多加1维；这里的c即为预测的类别总数（应该是吧）
        ############################################################################
        # 框中这部分还是有问题，得看源代码理解下
        # 下两行操作我懂，通道加1维，并复制
        consistent_idxs_x = consistent_idxs.unsqueeze(1).repeat(1, 5, 1, 1, 1)  ###B*C*H*W
        inconsistent_idxs_x = inconsistent_idxs.unsqueeze(1).repeat(1, 5, 1, 1, 1)  ###B*C*H*W

        # 纳了闷儿了，这样如果两张图像的预测概率都一样，那不是会被覆盖掉吗？？？
        score_t_aug_pos = torch.where(consistent_idxs_x, aug_pred, score_t_aug_pos)  ###B*C*H*W
        score_t_aug_neg = torch.where(inconsistent_idxs_x, aug_pred, score_t_aug_neg)  ####B*C*H*W
        ############################################################################

    correct_mask, incorrect_mask = correct_mask > (q_len // 2), incorrect_mask > (q_len // 2)
    # B*H*W得到像素总数目，correct_mask全部求和得到预测正确的数目
    correct_ratio = (correct_mask).sum().item() / \
                    (batch_img_shape[0] * batch_img_shape[2] * batch_img_shape[3] * batch_img_shape[4])
    incorrect_ratio = (incorrect_mask).sum().item() / \
                      (batch_img_shape[0] * batch_img_shape[2] * batch_img_shape[3] * batch_img_shape[4])

    if correct_ratio > 0.0:
        probs_t_pos = F.softmax(score_t_aug_pos, dim=1)
        # probs_t_pos_sum = prob_entropyloss_sel(score_t_og_ori)
        probs_t_pos_sum = torch.sum(probs_t_pos* (torch.log(probs_t_pos + 1e-12)), 1)
        # print(torch.masked_select(probs_t_pos_sum, correct_mask).shape)
        loss_cent_correct = 1 * correct_ratio * torch.mean(torch.masked_select(probs_t_pos_sum, correct_mask))
    else:
        loss_cent_correct = 0

    if incorrect_ratio > 0.0:
        probs_t_neg = F.softmax(score_t_aug_neg, dim=1)
        # probs_t_neg_sum = prob_entropyloss_sel(score_t_og_ori)
        probs_t_neg_sum = torch.sum(probs_t_neg * (torch.log(probs_t_neg + 1e-12)), 1)
        loss_cent_incorrect = 1 * incorrect_ratio * -torch.mean(torch.masked_select(probs_t_neg_sum, incorrect_mask))
        # print(torch.masked_select(probs_t_neg_sum, incorrect_mask).shape)
    else:
        loss_cent_incorrect = 0

    return loss_cent_correct + loss_cent_incorrect


def style_dist(means0, means1, vars0, vars1):
    dist = 0
    for m0, m1, v0, v1 in zip(means0, means1, vars0, vars1):
        dist += (torch.norm(m0.detach().cpu() - m1.detach().cpu(), p=2) + torch.norm(v0.detach().cpu() - v1.detach().cpu(), p=2))
    return dist