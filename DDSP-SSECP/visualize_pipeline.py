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
    plt.figure(figsize=(4, 4))
    plt.imshow(img_data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def save_3d_volume(volume, save_dir, name_prefix, is_feature=False, is_mask=False):
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
             # Assume it's a single slice or squeezed 3D volume with D=1
             # If it's a mask like M_alpha which is (H, W) for a single slice batch?
             # Based on error, M_alpha came in as (128, 128).
             # Let's reshape it to (1, H, W) to treat it as a volume with depth 1
             volume = volume[np.newaxis, ...]
        D, H, W = volume.shape
        for d in range(D):
            slice_dir = os.path.join(save_dir, f'slice_{d:03d}')
            os.makedirs(slice_dir, exist_ok=True)
            save_path = os.path.join(slice_dir, f'{name_prefix}.png')
            
            if is_mask:
                # Mask 通常是离散值，不进行归一化，保持原始值展示
                save_slice(volume[d, :, :], save_path, cmap='gray', vmin=0, vmax=1)
            else:
                # 图像数据，正常显示
                save_slice(volume[d, :, :], save_path, cmap='gray')

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
    # 假设权重文件命名格式固定，如果用户提供的是具体文件路径，需要自行调整
    if os.path.isdir(checkpoint_dir):
         # 尝试自动寻找 best 或特定 epoch
         enc_path = os.path.join(checkpoint_dir, 'ec_epoch_best.pth')
         seg_path = os.path.join(checkpoint_dir, 'seg_epoch_best.pth')
         enc_ema_path = os.path.join(checkpoint_dir, 'ec_ema_epoch_best.pth')
         seg_ema_path = os.path.join(checkpoint_dir, 'seg_ema_epoch_best.pth')
    else:
         # 简单的假设：用户传入的是 ec_epoch_best.pth 的路径，我们需要推断其他路径
         # 这里为了稳健，还是建议传入文件夹路径。
         # 如果用户传入具体文件，我们假设是 enc 的文件，并尝试推断其他的
         base_dir = os.path.dirname(checkpoint_dir)
         base_name = os.path.basename(checkpoint_dir)
         # 简单的替换逻辑，实际情况可能需要调整
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
                                    sigma=(10., 13.)) # 传入device

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
        srsimg, srsimg_r, srslabel = batch_srs
        tarimg, tarimg_r, _ = batch_tar
        
        # Get patient ID (based on file list index)
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
        # 处理 Label 格式
        srslabel_np = srslabel_aug.cpu().numpy()[0][0]
        srslabel_onehot = torch.from_numpy(
            to_categorical(srslabel_np, num_classes=args.num_classes)[np.newaxis, :, :, :, :]
        ).to(device)
        
        # 生成 CP 图像 (Copy-Paste)
        img_boxes = context_mask_mul(img_size=srsimg.shape[2:], box_num=1, min_mask_ratio=1/6, max_mask_ratio=1/3)
        img_boxes = img_boxes.to(device) # M_alpha
        
        srs_img_cp = srsimg * (1 - img_boxes) + tarimg * img_boxes
        tar_img_cp = tarimg * (1 - img_boxes) + srsimg * img_boxes
        srs_img_r_cp = srsimg_r * (1 - img_boxes) + tarimg * img_boxes
        tar_img_r_cp = tarimg_r * (1 - img_boxes) + srsimg * img_boxes
        
        # 6. 前向传播 - 学生模型 (获取8个特征和8个预测)
        inputs_student = {
            'srs_img': srsimg, 'srs_img_r': srsimg_r,
            'srs_img_cp': srs_img_cp, 'srs_img_r_cp': srs_img_r_cp,
            'tar_img': tarimg, 'tar_img_r': tarimg_r,
            'tar_img_cp': tar_img_cp, 'tar_img_r_cp': tar_img_r_cp
        }
        
        features = {}
        preds_student = {}
        
        # print("Running student model inference...")
        with torch.no_grad():
            for name, img in inputs_student.items():
                # 编码器
                feat = enc(img)
                features[name] = feat
                # 解码器
                pred = seg(feat)
                preds_student[name] = pred

        # 7. 前向传播 - 教师模型 (获取3个预测和伪标签)
        # print("Running teacher model inference...")
        threshold = 0.95
        with torch.no_grad():
            # 7.1 tar_img
            prob_ulb_x_w = seg_ema(enc_ema(tarimg))
            prob, pseudo_label = torch.max(prob_ulb_x_w, dim=1)
            pseudo_label = pseudo_label[0] # (D, H, W)
            tar_mask = (prob > threshold)[0].float()
            
            # 7.2 srs_img_cp
            prob_w_lu = seg_ema(enc_ema(srs_img_cp))
            _, pseudo_label_src_cp = torch.max(prob_w_lu, dim=1)
            pseudo_label_src_cp = pseudo_label_src_cp[0]
        
        # 7.3 tar_img_cp
        prob_w_ul = seg_ema(enc_ema(tar_img_cp))
        _, pseudo_label_tar_cp = torch.max(prob_w_ul, dim=1)
        pseudo_label_tar_cp = pseudo_label_tar_cp[0]
        
        # 7.4 伪标签融合 (pseudo_label_w)
        # srs_label_indices 处理 (GT for source regions)
        srs_label_indices = torch.argmax(srslabel_onehot, dim=1)[0]
        box_mask = img_boxes.squeeze()
        
        # 修正 CP 图伪标签 (源域区域用 GT)
        # srs_img_cp: 背景是源域 (box==0)
        source_region = (box_mask == 0)
        pseudo_label_src_cp[source_region] = srs_label_indices[source_region]
        
        # tar_img_cp: 前景是源域 (box==1)
        source_region_tar = (box_mask == 1)
        pseudo_label_tar_cp[source_region_tar] = srs_label_indices[source_region_tar]
        
        # 融合
        pseudo_label_w = pseudo_label_tar_cp * (1 - img_boxes.squeeze()) + pseudo_label_src_cp * img_boxes.squeeze()
        
        # 最终掩码 (mask_w)
        # 这里简化处理，不完全复现 tar_mask_cp 的生成细节，直接关注最终融合逻辑
        # 主要是可视化 pseudo_label_w 和 mask_w
        # mask_w 逻辑：ensemble 校验
        ensemble = (pseudo_label_w == pseudo_label).float() * tar_mask
        mask_w = ensemble.clone() # 简单起见，仅展示 ensemble 后的 mask

        # 8. 保存结果
        save_root = os.path.join(args.output_dir, patient_id)
        # print(f"Saving results to {save_root}...")
        
        # 8.1 保存输入图像 (8个)
        for name, img in inputs_student.items():
            save_3d_volume(img[0].cpu().numpy(), save_root, f"Input_{name}")
            
        # 8.2 保存源域 GT
        print(f"Saving results for {patient_id}...") 
        save_3d_volume(srslabel_aug[0, 0].cpu().numpy().astype(float), save_root, "Input_srs_GT", is_mask=True)
        
        # 8.3 保存 M_alpha
        save_3d_volume(img_boxes.cpu().numpy().astype(float), save_root, "Input_M_alpha", is_mask=True)

        # 8.4 保存中间特征 (8个)
        for name, feat in features.items():
            save_3d_volume(feat[0].cpu().numpy(), save_root, f"Feat_{name}", is_feature=True)
        
        # 8.5 保存学生模型输出 (16个: 8个Mask + 8个ProbMap，这里只保存Mask)
        for name, pred in preds_student.items():
            pred_mask = torch.argmax(pred, dim=1).float()
            save_3d_volume(pred_mask[0].cpu().numpy(), save_root, f"StudentPred_{name}", is_mask=True)
            
        # 8.6 保存教师模型输出 (3个)
        save_3d_volume(pseudo_label.cpu().numpy().astype(float), save_root, "TeacherPred_tar_img", is_mask=True)
        save_3d_volume(pseudo_label_src_cp.cpu().numpy().astype(float), save_root, "TeacherPred_srs_img_cp", is_mask=True)
        save_3d_volume(pseudo_label_tar_cp.cpu().numpy().astype(float), save_root, "TeacherPred_tar_img_cp", is_mask=True)
        
        # 8.7 保存融合结果
        save_3d_volume(pseudo_label_w.cpu().numpy().astype(float), save_root, "Final_PseudoLabel_w", is_mask=True)
        save_3d_volume(mask_w.cpu().numpy(), save_root, "Final_Mask_w", is_mask=True)
    
    print("Visualization completed!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--A_root', default="./Pro128/BIDMC", help='Path to source domain dataset')
    parser.add_argument('--B_root', default="./Pro128/HK", help='Path to target domain dataset')
    parser.add_argument('--checkpoint_path', required=True, help='Path to the model checkpoint directory or file')
    parser.add_argument('--output_dir', default="./vis", help='Output directory for visualization')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (Pro128 is 2)')
    parser.add_argument('--srs_rmmax', type=int, default=10)
    parser.add_argument('--tar_rmmax', type=int, default=10)
    
    parser.add_argument('--fold_num', type=int, default=2)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--direction', default="B2A")
    parser.add_argument('--permutationA', default=None)
    parser.add_argument('--permutationB', default=None)
    
    args = parser.parse_args()
    
    run_visualization(args)
