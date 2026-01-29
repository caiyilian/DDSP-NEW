import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from models.network import Seg, UNet_base
from utils.dataloader_origin import Dataset3D_remap as TrainDataset
from utils.utils import context_mask_mul, K_fold_data_gen, K_fold_file_gen
from utils.Transform_self_origin import SpatialTransform
from torch.utils.data import DataLoader
from tqdm import tqdm

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

def save_slice(img_data, save_path, cmap='gray', vmin=None, vmax=None):
    """
    保存单张切片图像
    img_data: 2D numpy array
    """
    plt.figure(figsize=(1, 1))
    plt.imshow(img_data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_3d_volume(volume, save_dir, name_prefix, is_feature=False, is_mask=False, num_classes=None):
    """
    保存3D体的所有切片
    volume: (D, H, W) or (C, D, H, W) numpy array
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if is_feature:
        # volume shape: (C, D, H, W)
        C, D, H, W = volume.shape
        for d in range(D):
            slice_dir = os.path.join(save_dir, f'slice_{d:03d}')
            os.makedirs(slice_dir, exist_ok=True)
            for c in range(C):
                save_path = os.path.join(slice_dir, f'{name_prefix}_feat{c}.png')
                save_slice(volume[c, d, :, :], save_path, cmap='jet')
    else:
        # volume shape: (D, H, W)
        if len(volume.shape) == 4 and volume.shape[0] == 1:
            volume = volume[0]
        if len(volume.shape) == 2:
             volume = volume[np.newaxis, ...]
        D, H, W = volume.shape
        
        # 处理 Mask 的灰度映射 (0-4 -> 0-255)
        if is_mask and num_classes is not None and num_classes > 1:
            # 等比例映射
            factor = 255.0 / (num_classes - 1)
            volume = volume * factor
            vmin = 0
            vmax = 255
        elif is_mask:
            # 如果没有提供 num_classes，或者只有1类（不应该发生），保持原样或默认 0-1
            vmin = 0
            vmax = 1
            if num_classes: # if num_classes provided but <=1 ?
                pass
            # 兼容旧逻辑，如果没有 num_classes 参数传入，假设是 0-1 mask
            if num_classes is None:
                vmin = 0
                vmax = 1
        else:
            vmin = None
            vmax = None

        for d in range(D):
            slice_dir = os.path.join(save_dir, f'slice_{d:03d}')
            os.makedirs(slice_dir, exist_ok=True)
            save_path = os.path.join(slice_dir, f'{name_prefix}.png')
            
            save_slice(volume[d, :, :], save_path, cmap='gray', vmin=vmin, vmax=vmax)

def run_visualization(args):
    # 1. 设置环境和设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 初始化模型
    print("Initializing models...")
    enc = UNet_base(input_channels=1).to(device)
    seg = Seg(out=args.num_classes).to(device)
    
    # 教师模型
    enc_ema = UNet_base(input_channels=1).to(device)
    seg_ema = Seg(out=args.num_classes).to(device)
    
    # 加载权重
    print(f"Loading weights from {args.checkpoint_path}...")
    checkpoint_dir = args.checkpoint_path
    if os.path.isdir(checkpoint_dir):
         enc_path = os.path.join(checkpoint_dir, 'ec_epoch_best.pth')
         seg_path = os.path.join(checkpoint_dir, 'seg_epoch_best.pth')
         enc_ema_path = os.path.join(checkpoint_dir, 'ec_ema_epoch_best.pth')
         seg_ema_path = os.path.join(checkpoint_dir, 'seg_ema_epoch_best.pth')
    else:
         enc_path = checkpoint_dir
         seg_path = checkpoint_dir.replace('ec_', 'seg_')
         enc_ema_path = checkpoint_dir.replace('ec_', 'ec_ema_')
         seg_ema_path = checkpoint_dir.replace('ec_', 'seg_ema_')

    try:
        enc.load_state_dict(torch.load(enc_path, map_location=device))
        seg.load_state_dict(torch.load(seg_path, map_location=device))
        enc_ema.load_state_dict(torch.load(enc_ema_path, map_location=device))
        seg_ema.load_state_dict(torch.load(seg_ema_path, map_location=device))
    except FileNotFoundError as e:
        print(f"Error loading weights: {e}")
        return

    enc.eval()
    seg.eval()
    enc_ema.eval()
    seg_ema.eval()
    
    # 3. 准备数据增强模块
    spatial_aug = SpatialTransform(do_rotation=True,
                                    angle_x=(-np.pi / 9, np.pi / 9),
                                    angle_y=(-np.pi / 9, np.pi / 9),
                                    angle_z=(-np.pi / 9, np.pi / 9),
                                    do_scale=True,
                                    scale_x=(0.75, 1.25),
                                    scale_y=(0.75, 1.25),
                                    scale_z=(0.75, 1.25),
                                    do_translate=True,
                                    trans_x=(-0.1, 0.1),
                                    trans_y=(-0.1, 0.1),
                                    trans_z=(-0.1, 0.1),
                                    do_shear=True,
                                    shear_xy=(-np.pi / 18, np.pi / 18),
                                    shear_xz=(-np.pi / 18, np.pi / 18),
                                    shear_yx=(-np.pi / 18, np.pi / 18),
                                    shear_yz=(-np.pi / 18, np.pi / 18),
                                    shear_zx=(-np.pi / 18, np.pi / 18),
                                    shear_zy=(-np.pi / 18, np.pi / 18),
                                    do_elastic_deform=True,
                                    alpha=(0., 512.),
                                    sigma=(10., 13.))

    # 4. 准备数据集
    print("Preparing dataset...")
    
    # Data splitting logic
    if args.permutationA:
        pA = np.load(args.permutationA)
        pA = pA.tolist()
    else:
        pA = list(range(len(os.listdir(args.A_root))))

    if args.permutationB:
        pB = np.load(args.permutationB)
        pB = pB.tolist()
    else:
        pB = list(range(len(os.listdir(args.B_root))))

    data_listA = np.array(sorted([os.path.join(args.A_root, x) for x in os.listdir(args.A_root)]))[pA].tolist()
    data_listA = K_fold_file_gen(data_listA, args.fold_num, is_shuffle=False)
    train_kA, valid_kA = K_fold_data_gen(data_listA, args.fold - 1, args.fold_num)

    data_listB = np.array(sorted([os.path.join(args.B_root, x) for x in os.listdir(args.B_root)]))[pB].tolist()
    data_listB = K_fold_file_gen(data_listB, args.fold_num, is_shuffle=False)
    train_kB, valid_kB = K_fold_data_gen(data_listB, args.fold - 1, args.fold_num)

    if args.direction == "A2B":
        train_srs = train_kA + valid_kA
        train_tar = train_kB
    elif args.direction == "B2A":
        train_srs = train_kB + valid_kB
        train_tar = train_kA
    
    print(f"Total Source Training Samples: {len(train_srs)}")
    print(f"Total Target Training Samples: {len(train_tar)}")

    trainsrs_dataset = TrainDataset(train_srs, rmmax=args.srs_rmmax, trans_type='Remap')
    traintar_dataset = TrainDataset(train_tar, rmmax=args.tar_rmmax, trans_type='Remap')

    dataloader_srs = DataLoader(trainsrs_dataset, batch_size=1, shuffle=False)
    dataloader_tar = DataLoader(traintar_dataset, batch_size=1, shuffle=False)

    # 5. 循环处理所有数据
    for idx, (batch_srs, batch_tar) in enumerate(zip(dataloader_srs, dataloader_tar)):
        pbar = tqdm(total=16, desc=f"Batch {idx+1}")
        srsimg, srsimg_r, srslabel = batch_srs
        tarimg, tarimg_r, _ = batch_tar
        
        srs_path = train_srs[idx]
        tar_path = train_tar[idx]
        patient_id = os.path.basename(srs_path)
        
        print(f"\nProcessing Pair {idx+1}: SRS={patient_id}, TAR={os.path.basename(tar_path)}")

        srsimg = srsimg.to(device)
        srsimg_r = srsimg_r.to(device)
        tarimg = tarimg.to(device)
        tarimg_r = tarimg_r.to(device)
        srslabel = srslabel.to(device)
        
        # 数据增强 (Spatial Augmentation)
        # mat, code_spa = spatial_aug.rand_coords(srsimg.shape[2:])
        # srsimg = spatial_aug.augment_spatial(srsimg, mat, code_spa)
        # srslabel_aug = spatial_aug.augment_spatial(srslabel, mat, code_spa, mode="nearest").int()
        # srsimg_r = spatial_aug.augment_spatial(srsimg_r, mat, code_spa)
        # tarimg = spatial_aug.augment_spatial(tarimg, mat, code_spa)
        # tarimg_r = spatial_aug.augment_spatial(tarimg_r, mat, code_spa)
        srslabel_aug = srslabel
        
        srslabel_np = srslabel_aug.cpu().numpy()[0][0]
        srslabel_onehot = torch.from_numpy(
            to_categorical(srslabel_np, num_classes=args.num_classes)[np.newaxis, :, :, :, :]
        ).to(device)
        # 生成 CP 图像
        img_boxes = context_mask_mul(img_size=srsimg.shape[2:], box_num=1, min_mask_ratio=1/6, max_mask_ratio=1/3)

        img_boxes = img_boxes.to(device) # M_alpha
        
        srs_img_cp = srsimg * (1 - img_boxes) + tarimg * img_boxes
        tar_img_cp = tarimg * (1 - img_boxes) + srsimg * img_boxes
        srs_img_r_cp = srsimg_r * (1 - img_boxes) + tarimg * img_boxes
        tar_img_r_cp = tarimg_r * (1 - img_boxes) + srsimg * img_boxes
        # 6. 前向传播 - 学生模型
        inputs_student = {
            'srs_img': srsimg, 'srs_img_r': srsimg_r,
            'srs_img_cp': srs_img_cp, 'srs_img_r_cp': srs_img_r_cp,
            'tar_img': tarimg, 'tar_img_r': tarimg_r,
            'tar_img_cp': tar_img_cp, 'tar_img_r_cp': tar_img_r_cp
        }
        
        features = {}
        preds_student = {}
        
        with torch.no_grad():
            for name, img in inputs_student.items():
                feat = enc(img)
                features[name] = feat
                pred = seg(feat)
                preds_student[name] = pred
        # 7. 前向传播 - 教师模型
        threshold = 0.95
        with torch.no_grad():
            # 7.1 tar_img
            prob_ulb_x_w = seg_ema(enc_ema(tarimg))
            prob, pseudo_label = torch.max(prob_ulb_x_w, dim=1) #pseudo_label unique: 0 1 2 3 4
            
            pseudo_label = pseudo_label[0]
            tar_mask = (prob > threshold)[0].float()
            
            # 7.2 srs_img_cp
            prob_w_lu = seg_ema(enc_ema(srs_img_cp))
            _, pseudo_label_src_cp = torch.max(prob_w_lu, dim=1)
            pseudo_label_src_cp = pseudo_label_src_cp[0]
            src_mask_cp = (prob_w_lu.max(dim=1)[0] > threshold)[0].float()
        
            # 7.3 tar_img_cp
            prob_w_ul = seg_ema(enc_ema(tar_img_cp))
            _, pseudo_label_tar_cp = torch.max(prob_w_ul, dim=1)
            pseudo_label_tar_cp = pseudo_label_tar_cp[0]
            tar_mask_cp = (prob_w_ul.max(dim=1)[0] > threshold)[0].float()
        
        # 7.4 伪标签融合与Mask生成
        srs_label_indices = torch.argmax(srslabel_onehot, dim=1)[0]
        box_mask = img_boxes.squeeze()
        
        # 保存替换前的原始伪标签 (Clone before modification)
        pseudo_label_src_cp_raw = pseudo_label_src_cp.clone()
        pseudo_label_tar_cp_raw = pseudo_label_tar_cp.clone()

        # 修正 CP 图伪标签与 Mask (源域区域用 GT)
        # srs_img_cp: 背景是源域 (box==0)
        source_region = (box_mask == 0)
        pseudo_label_src_cp[source_region] = srs_label_indices[source_region]
        src_mask_cp[source_region] = 1.0
        
        # tar_img_cp: 前景是源域 (box==1)
        source_region_tar = (box_mask == 1)
        pseudo_label_tar_cp[source_region_tar] = srs_label_indices[source_region_tar]
        tar_mask_cp[source_region_tar] = 1.0
        
        pseudo_label_w = pseudo_label_tar_cp * (1 - img_boxes.squeeze()) + pseudo_label_src_cp * img_boxes.squeeze()
        
        # 计算 mask_w
        # mask_w 逻辑：ensemble 校验
        # 原始代码中 mask_w = tar_mask_cp * (1 - box) + src_mask_cp * box
        # 然后 mask_w[ensemble == 0] = 0
        mask_w_pre = tar_mask_cp * (1 - img_boxes.squeeze()) + src_mask_cp * img_boxes.squeeze()
        ensemble = (pseudo_label_w == pseudo_label).float() * tar_mask
        mask_w = mask_w_pre.clone()
        mask_w[ensemble == 0] = 0
        
        # 生成其余分割损失 Mask
        # src_r_mask & tar_r_mask
        # srs_img_r_cp = srs_r * (1 - box) + tar * box
        # src_r_mask: if box==0 (srs_r), trust(1); if box==1 (tar), trust tar_mask
        src_r_mask = tar_mask.clone()
        src_r_mask[img_boxes.squeeze() == 0] = 1
        
        # tar_img_r_cp = tar_r * (1 - box) + srs * box
        # tar_r_mask: if box==1 (srs), trust(1); if box==0 (tar_r), trust tar_mask
        tar_r_mask = tar_mask.clone()
        tar_r_mask[img_boxes.squeeze() == 1] = 1
        
        # 生成特征损失 Mask (4个)
        # 1. srs_mask (GT > 0)
        srs_mask = (srslabel_np > 0).astype(float) # srslabel_np is numpy, srslabel_aug is tensor
        srs_mask_tensor = torch.from_numpy(srs_mask).to(device).float()
        
        # 2. tar_mask_semantic (pseudo_label & tar_mask)
        # semantic mask 意味着既要有高置信度，又要有语义内容(>0)? 
        # 代码中: self.tar_mask_semantic = (self.pseudo_label & self.tar_mask.bool()).bool()
        # 这里 pseudo_label 是 long tensor, 包含类别索引 0-4.
        # 位运算 & 对 long 和 bool 可能会有问题，除非 pseudo_label 被视作 bool (即 > 0)
        # 让我们仔细看 train_cp_all_merge_pseudo_mmwhs.py: 
        # self.pseudo_label = self.pseudo_label[0] (D,H,W) long
        # self.tar_mask = (prob > threshold)[0].float() -> .bool()
        # (self.pseudo_label & self.tar_mask.bool())
        # Python位运算: 非0整数与bool True(1) 进行 &，结果是非0整数?
        # 如果 pseudo_label=0 (背景), 0 & 1 = 0 (False)
        # 如果 pseudo_label=2, 2 & 1 = 0 (False) ?? 2 is 10 binary. 1 is 01. 10 & 01 = 00.
        # 等等，PyTorch 的 & 运算符对于 ByteTensor/BoolTensor 是 element-wise AND.
        # 如果 pseudo_label 是 LongTensor, 它执行按位与.
        # 类别 1 (01), 2 (10), 3 (11), 4 (100).
        # tar_mask.bool() is 0 or 1.
        # 如果 tar_mask=1. 
        # 1 & 1 = 1. 2 & 1 = 0. 3 & 1 = 1. 4 & 1 = 0.
        # 这看起来不对劲。除非原意是 (pseudo_label > 0) & tar_mask.
        # 或者 pseudo_label 已经是 one-hot? 在 compute_fea_loss 之前，pseudo_label 还是 index map.
        # 但在 compute_fea_loss 之前有:
        # self.pseudo_label = torch.from_numpy(to_categorical(...))
        # 变成了 one-hot (C, D, H, W).
        # 然后: tar_cs_loss = crosschannel_sim(latent_b[:, :, tar_mask_semantic], ...)
        # 如果 tar_mask_semantic 是 (C, D, H, W) 或者是 spatial mask (D, H, W)?
        # 训练代码中: self.tar_mask_semantic = (self.pseudo_label & self.tar_mask.bool()).bool()
        # 这一行是在 pseudo_label 被转为 one-hot *之前* 还是 *之后*?
        # 让我们回溯代码:
        # Line 205: self.tar_mask_semantic = (self.pseudo_label & self.tar_mask.bool()).bool()
        # Line 266: self.pseudo_label = torch.from_numpy(... onehot ...)
        # 所以是在 one-hot 之前。此时 pseudo_label 是 (D, H, W) 的 index map.
        # 那么 `pseudo_label & tar_mask.bool()` 这种写法在 PyTorch 中对 LongTensor 和 BoolTensor 的行为是什么?
        # 这是一个潜在的 bug 或者是我理解偏差。
        # 如果作者意图是 "Foreground check"，应该是 `(pseudo_label > 0) & tar_mask`.
        # 但如果是位运算:
        # 1 (001) & 1 = 1
        # 2 (010) & 1 = 0
        # 3 (011) & 1 = 1
        # 4 (100) & 1 = 0
        # 这会导致偶数类别被过滤掉！这在医学图像分割中如果是多类，这显然是错的。
        # 但也许作者只关心二分类? Pro128 是 2 类 (0, 1). 1 & 1 = 1. 0 & 1 = 0. 没问题.
        # 现在 MMWHS 是 5 类.
        # 如果直接照搬代码，可能会丢失偶数类别 (2, 4) 的特征监督。
        # 用户让我 "不要改代码"，但是这里是 "弄一个适用于 mmwhs 的"。
        # 我应该照搬逻辑，还是修复它?
        # 考虑到用户说 "不要改我的进度条设置... 改完之后仔细检查"，这暗示他希望代码正确运行。
        # 但如果原训练代码就是这样写的，通过 & 运算，那复现就应该保持一致?
        # 或者 tar_mask 其实也是 LongTensor? `(prob > threshold)[0].float()` -> float.
        # 让我们假设这是为了提取前景特征。
        # 在多类情况下，通常特征一致性是针对所有前景或者特定类别。
        # 为了稳妥，我将完全复制逻辑: `(pseudo_label.long() & tar_mask.long().bool().long()).bool()`
        # 实际上 PyTorch 中 `long & bool` 会报错吗? 或者自动广播?
        # 为了可视化，我应该展示 "Mask"。
        # 我会严格按照训练代码的逻辑行事，并在注释中指出这一点。
        # 但等等，tar_mask 在训练代码里被转换了: `self.tar_mask = (prob > threshold)[0].float()`
        # 然后 `(self.pseudo_label & self.tar_mask.bool())`
        # 让我们用代码实现这个逻辑。
        
        tar_mask_bool = tar_mask.bool()
        tar_mask_semantic = (pseudo_label.long() & tar_mask_bool.long()).bool() # bitwise AND
        # 注意: 如果 pseudo_label > 1，这个逻辑会很奇怪。
        # 但为了复现 mask，我先这样写。如果全是 0，那就是这个逻辑在多类下有问题。
        # 不过，为了可视化更有意义，如果这是为了 filter features，也许应该显示 mask 本身。
        
        src_mask_cp_bool = src_mask_cp.bool()
        src_mask_cp_semantic = (pseudo_label_src_cp.long() & src_mask_cp_bool.long()).bool()
        
        tar_mask_cp_bool = tar_mask_cp.bool()
        tar_mask_cp_semantic = (pseudo_label_tar_cp.long() & tar_mask_cp_bool.long()).bool()

        pbar.update(1)
        # 8. 保存结果
        save_root = os.path.join(args.output_dir, patient_id)
        pbar.update(1)
        # 8.1 保存输入图像
        for name, img in inputs_student.items():
            save_3d_volume(img[0].cpu().numpy(), save_root, f"Input_{name}")
        pbar.update(1)
        # 8.2 保存源域 GT (Mask)
        print(f"Saving results for {patient_id}...") 
        save_3d_volume(srslabel_aug[0, 0].cpu().numpy().astype(float), save_root, "Input_srs_GT", is_mask=True, num_classes=args.num_classes)
        
        # 8.3 保存 M_alpha (Mask, 0/1)
        save_3d_volume(img_boxes.cpu().numpy().astype(float), save_root, "Input_M_alpha", is_mask=True, num_classes=2)
        pbar.update(1)
        # 8.4 保存中间特征
        for name, feat in features.items():
            pbar.update(1)
            save_3d_volume(feat[0].cpu().numpy(), save_root, f"Feat_{name}", is_feature=True)
        pbar.update(1)
        # 8.5 保存学生模型输出 (Mask)
        for name, pred in preds_student.items():
            pred_mask = torch.argmax(pred, dim=1).float()
            save_3d_volume(pred_mask[0].cpu().numpy(), save_root, f"StudentPred_{name}", is_mask=True, num_classes=args.num_classes)
        pbar.update(1)
        # 8.6 保存教师模型输出 (Mask)
        save_3d_volume(pseudo_label.cpu().numpy().astype(float), save_root, "TeacherPred_tar_img", is_mask=True, num_classes=args.num_classes)
        
        # 保存原始伪标签 (替换前)
        save_3d_volume(pseudo_label_src_cp_raw.cpu().numpy().astype(float), save_root, "TeacherPred_srs_img_cp_raw", is_mask=True, num_classes=args.num_classes)
        save_3d_volume(pseudo_label_tar_cp_raw.cpu().numpy().astype(float), save_root, "TeacherPred_tar_img_cp_raw", is_mask=True, num_classes=args.num_classes)
        
        # 保存最终伪标签 (替换后)
        save_3d_volume(pseudo_label_src_cp.cpu().numpy().astype(float), save_root, "TeacherPred_srs_img_cp", is_mask=True, num_classes=args.num_classes)
        save_3d_volume(pseudo_label_tar_cp.cpu().numpy().astype(float), save_root, "TeacherPred_tar_img_cp", is_mask=True, num_classes=args.num_classes)
        pbar.update(1)
        # 8.7 保存融合结果 (Mask)
        save_3d_volume(pseudo_label_w.cpu().numpy().astype(float), save_root, "Final_PseudoLabel_w", is_mask=True, num_classes=args.num_classes)
        pbar.update(1)
        
        # 8.8 保存分割损失 Mask (6个)
        # 1. srs_label (GT) - 已保存 (Input_srs_GT)
        # 2. mask_w
        save_3d_volume(mask_w.cpu().numpy(), save_root, "LossMask_Seg_mask_w", is_mask=True, num_classes=2)
        # 3. src_r_mask
        save_3d_volume(src_r_mask.cpu().numpy(), save_root, "LossMask_Seg_src_r_mask", is_mask=True, num_classes=2)
        # 4. tar_r_mask
        save_3d_volume(tar_r_mask.cpu().numpy(), save_root, "LossMask_Seg_tar_r_mask", is_mask=True, num_classes=2)
        # 5. src_mask_cp
        save_3d_volume(src_mask_cp.cpu().numpy(), save_root, "LossMask_Seg_src_mask_cp", is_mask=True, num_classes=2)
        # 6. tar_mask_cp
        save_3d_volume(tar_mask_cp.cpu().numpy(), save_root, "LossMask_Seg_tar_mask_cp", is_mask=True, num_classes=2)
        
        # 8.9 保存特征损失 Mask (4个)
        # 1. srs_mask
        save_3d_volume(srs_mask_tensor.cpu().numpy(), save_root, "LossMask_Feat_srs_mask", is_mask=True, num_classes=2)
        # 2. tar_mask_semantic
        save_3d_volume(tar_mask_semantic.float().cpu().numpy(), save_root, "LossMask_Feat_tar_mask_semantic", is_mask=True, num_classes=2)
        # 3. src_mask_cp_semantic
        save_3d_volume(src_mask_cp_semantic.float().cpu().numpy(), save_root, "LossMask_Feat_src_mask_cp_semantic", is_mask=True, num_classes=2)
        # 4. tar_mask_cp_semantic
        save_3d_volume(tar_mask_cp_semantic.float().cpu().numpy(), save_root, "LossMask_Feat_tar_mask_cp_semantic", is_mask=True, num_classes=2)

    
    print("Visualization completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--A_root', default="./mmwhs_96/ct_debugging96", help='Path to source domain dataset')
    parser.add_argument('--B_root', default="./mmwhs_96/mr_debugging96", help='Path to target domain dataset')
    parser.add_argument('--checkpoint_path', required=True, help='Path to the model checkpoint directory or file')
    parser.add_argument('--output_dir', default="./vis_mmwhs", help='Output directory for visualization')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes (MMWHS is 5)')
    parser.add_argument('--srs_rmmax', type=int, default=10)
    parser.add_argument('--tar_rmmax', type=int, default=10)
    
    parser.add_argument('--fold_num', type=int, default=2)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--direction', default="B2A")
    parser.add_argument('--permutationA', default=None)
    parser.add_argument('--permutationB', default=None)
    
    args = parser.parse_args()
    
    run_visualization(args)
