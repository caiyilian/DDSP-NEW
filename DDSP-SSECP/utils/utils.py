import csv
import math
import numpy as np
import random
import torch
import torch.nn.functional as F

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class LogWriter(object):
    def __init__(self, name, head):
        self.name = name+'.csv'
        with open(self.name, 'w', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(head)
            f.close()

    def writeLog(self, dict):
        with open(self.name, 'a', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(dict)
            f.close()

def dice(pre, gt):
    tmp = pre + gt
    a = np.sum(np.where(tmp == 2, 1, 0))
    b = np.sum(pre)
    c = np.sum(gt)
    dice = (2*a)/(b+c+1e-6)
    return dice

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((num_classes, n))
    categorical[y, np.arange(n)] = 1
    output_shape = (num_classes,) + input_shape
    categorical = np.reshape(categorical, output_shape)
    return categorical

def EMA(model_A, model_B, alpha=0.999):
    for param_B, param_A in zip(model_B.parameters(), model_A.parameters()):
        param_A.data = alpha*param_A.data + (1-alpha)*param_B.data
    return model_A

def adjust_learning_rate(optimizer, epoch, epochs, lr, schedule, is_cos=False):
    if is_cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    else:  # stepwise lr schedule
        for milestone in schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def K_fold_file_gen(data_list, k, is_shuffle=False):
    assert k > 1
    length = len(data_list)
    fold_size = length // k
    data_numpy = np.array(data_list)
    if is_shuffle:
        index = [i for i in range(0, length)]
        random.shuffle(index)
        data_numpy = data_numpy[index]

    outputlist = list()

    for i in range(0, k):
        idx = slice(i * fold_size, (i + 1) * fold_size)
        if i == k-1:
            idx = slice(i * fold_size, length)
        i_list = data_numpy[idx].tolist()

        outputlist.append(i_list)

    return tuple(outputlist)


def K_fold_data_gen(data_list, i, k):

    valid_list = data_list[i]
    train_list = list()
    for littleseries in range(0, i):
        train_list = train_list + data_list[littleseries]
    for littleseries in range(i + 1, k):
        train_list = train_list + data_list[littleseries]


    return train_list, valid_list

def selfchannel_sim(fe):
    # 计算通道相似度矩阵
    x = fe[0]
    y = fe[0].permute(1, 0)

    x_norm = F.normalize(x, p=2, dim=1)  
    y_norm = F.normalize(y, p=2, dim=0)

    selfdiffusion = torch.matmul(x_norm, y_norm)
    selfdiffusion = selfdiffusion - selfdiffusion.min() + 1e-8
    selfdiffusion = (selfdiffusion + selfdiffusion.permute(1, 0)) / 2.0 
    selfdiffusion /= selfdiffusion.sum(dim=1)

    return selfdiffusion

def selfchannel_loss(srs, tar):
    # 目的：不同模态的同一个通道，其应当相似
    srs_diffusion = selfchannel_sim(srs)
    tar_diffusion = selfchannel_sim(tar)

    loss = torch.nn.L1Loss(reduction="mean")
    kl_loss = loss(srs_diffusion, tar_diffusion)
    return kl_loss

def crosschannel_sim(srs, tar):

    x = srs[0]
    y = tar[0]
    similarity = F.cosine_similarity(x, y, dim=1)
    return -torch.mean(similarity)

def print_network_para(model):
    print("------------------------------------------")
    print("Network Architecture of Model:")
    num_para = 0
    for name, param in model.named_parameters():
        num_mul = 1
        for x in param.SIZE():
            num_mul *= x
        num_para += num_mul

    print("Number of trainable parameters {0} in Model".format(num_para))
    print("------------------------------------------")

def sigmoid_rampup(current, low_length, max_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if max_length == 0:
        return 1.0
    else:
        current = np.clip(current, low_length, max_length)
        if current == low_length:
            return 0
        else:
            phase = 1.0 - (current - low_length) / (max_length - low_length)
            return float(np.exp(-5.0 * phase * phase))


def obtain_multi_cutmix_box_3D(img_size, p=1, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3, box_num=1):
    mask = torch.zeros(img_size).cuda()
    if random.random() > p:
        return mask

    w, h, d = img_size

    size = np.random.uniform(size_min, size_max) * img_size[0] * img_size[1] * img_size[2]
    for _ in range(box_num):
        while True:
            ratio = np.random.uniform(ratio_1, ratio_2)
            cutmix_w = int(np.cbrt(size) / ratio)
            cutmix_h = int(np.cbrt(size))
            cutmix_d = int(np.cbrt(size) * ratio)
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            z = np.random.randint(0, d)

            if x + cutmix_w <= img_size[0] and y + cutmix_h <= img_size[1] and z + cutmix_d <= img_size[2]:
                break

        mask[y:y + cutmix_h, x:x + cutmix_w, z:z + cutmix_d] = 1

    return mask


def context_mask(img_size, mask_ratio):
    img_x, img_y, img_z = img_size
    mask = torch.zeros(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x * mask_ratio), int(img_y * mask_ratio), int(
        img_z * mask_ratio)
    w = np.random.randint(0, img_x - patch_pixel_x)
    h = np.random.randint(0, img_y - patch_pixel_y)
    z = np.random.randint(0, img_z - patch_pixel_z)
    mask[w:w + patch_pixel_x, h:h + patch_pixel_y, z:z + patch_pixel_z] = 1

    return mask.long()


def context_mask_mul(img_size, min_mask_ratio=1/6, max_mask_ratio=1/3, box_num=1):
    img_x, img_y, img_z = img_size
    mask = torch.zeros(img_x, img_y, img_z).cuda()
    for _ in range(box_num):
        mask_ratio = np.random.uniform(min_mask_ratio, max_mask_ratio)
        patch_pixel_x, patch_pixel_y, patch_pixel_z = (
            int(img_x * mask_ratio), int(img_y * mask_ratio), int(img_z * mask_ratio))
        w = np.random.randint(0, img_x - patch_pixel_x)
        h = np.random.randint(0, img_y - patch_pixel_y)
        z = np.random.randint(0, img_z - patch_pixel_z)
        mask[w:w + patch_pixel_x, h:h + patch_pixel_y, z:z + patch_pixel_z] = 1

    return mask.long()

def update_ema_variables(model, ema_model, alpha, global_step):
    # teacher network: ema_model
    # student network: model
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

# ==================================================================================
# Added for PACCA (Prototype-Anchored Cross-domain Contrastive Alignment)
# ==================================================================================

class PrototypeBank(object):
    def __init__(self, num_classes, feature_dim, device, momentum=0.9):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.device = device
        self.momentum = momentum
        # Initialize prototypes
        self.prototypes = torch.zeros(num_classes, feature_dim).to(device)
        self.initialized = torch.zeros(num_classes).bool().to(device)

    def update(self, features, labels):
        # features: (B, F, D, H, W)
        # labels: (B, C, D, H, W) one-hot
        
        # Transpose features to (B, D, H, W, F)
        features = features.permute(0, 2, 3, 4, 1)
        
        for c in range(self.num_classes):
            # Get mask for class c
            # labels is one-hot (B, C, D, H, W)
            mask = labels[:, c, ...] == 1
            if mask.sum() > 0:
                # Calculate mean feature for this class
                class_features = features[mask]
                mean_feature = class_features.mean(dim=0)
                mean_feature = F.normalize(mean_feature, p=2, dim=0)
                
                if self.initialized[c]:
                    self.prototypes[c] = self.momentum * self.prototypes[c] + (1 - self.momentum) * mean_feature
                else:
                    self.prototypes[c] = mean_feature
                    self.initialized[c] = True
        
        # Normalize prototypes
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

def compute_centroids(features, labels, mask=None):
    # features: (B, F, D, H, W)
    # labels: (B, D, H, W) indices, or (B, C, D, H, W) one-hot
    # mask: (B, D, H, W) binary mask for valid pixels (e.g., confidence mask)
    
    B, F, D, H, W = features.shape
    features = features.permute(0, 2, 3, 4, 1).reshape(-1, F) # (N, F)
    
    if labels.dim() == 5: # One-hot (B, C, D, H, W)
         labels = torch.argmax(labels, dim=1) # (B, D, H, W)
    
    labels = labels.reshape(-1) # (N)
    
    if mask is not None:
        mask = mask.reshape(-1)
        features = features[mask]
        labels = labels[mask]
    
    centroids = {}
    classes = torch.unique(labels)
    
    for c in classes:
        c = c.item()
        c_features = features[labels == c]
        if c_features.shape[0] > 0:
            centroid = c_features.mean(dim=0)
            centroids[c] = F.normalize(centroid, p=2, dim=0)
            
    return centroids

def prototype_contrastive_loss(centroids, prototypes, tau=0.1):
    # centroids: dict {class_idx: feature_vector}
    # prototypes: Tensor (Num_Classes, F)
    
    loss = 0
    count = 0
    
    for c, centroid in centroids.items():
        if c >= prototypes.shape[0]: continue # Safety check
        
        # Positive pair: centroid and prototype[c]
        pos_sim = torch.sum(centroid * prototypes[c]) / tau
        
        # Negative pairs: centroid and prototype[j]
        all_sims = torch.matmul(prototypes, centroid) / tau # (Num_Classes)
        
        # Loss
        # -log( exp(pos) / sum(exp(all)) )
        loss += -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(all_sims)))
        count += 1
        
    if count > 0:
        return loss / count
    else:
        return torch.tensor(0.0).to(prototypes.device)

def structural_consistency_loss(centroids, prototypes):
    # centroids: dict {class_idx: feature_vector}
    # prototypes: Tensor (Num_Classes, F)
    
    present_classes = sorted(list(centroids.keys()))
    
    # Filter valid classes (ensure they are within prototype range)
    present_classes = [c for c in present_classes if c < prototypes.shape[0]]
    if len(present_classes) < 2:
        return torch.tensor(0.0).to(prototypes.device)
    
    indices = torch.tensor(present_classes).to(prototypes.device)
    
    # Source Gram (subset)
    sub_prototypes = prototypes[indices] # (K, F)
    gram_src = torch.matmul(sub_prototypes, sub_prototypes.t()) # (K, K)
    
    # Target Gram
    sub_centroids = torch.stack([centroids[c] for c in present_classes]) # (K, F)
    gram_tar = torch.matmul(sub_centroids, sub_centroids.t()) # (K, K)
    
    return F.mse_loss(gram_tar, gram_src)

if __name__ == '__main__':
    import torch
    x = torch.randn(1, 5, 50)
    y = torch.randn(1, 5, 50)
    sim1 = selfchannel_loss(x, y)
