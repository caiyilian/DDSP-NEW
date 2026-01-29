import os
from datetime import datetime
import torch
from torch import nn
from models.network import Seg, UNet_base
from utils.STN import SpatialTransformer
from utils.Transform_self_origin import SpatialTransform
from utils.dataloader_origin import Dataset3D_remap as TrainDataset
from utils.dataloader_origin import Dataset3D as TestDataset
from torch.utils.data import DataLoader
from utils.losses import dice_loss, prob_entropyloss
from utils.utils import (
    AverageMeter, LogWriter, K_fold_data_gen, K_fold_file_gen, dice,
    selfchannel_loss, crosschannel_sim, context_mask_mul, update_ema_variables,
    PrototypeBank, compute_centroids, prototype_contrastive_loss, structural_consistency_loss)
from utils import ramps
import numpy as np
import surface_distance as surfdist

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def crt_file(path):
    os.makedirs(path, exist_ok=True)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


class DDSPSeg(object):
    def __init__(self, args=None):
        super(DDSPSeg, self).__init__()

        self.fold_num = args.fold_num
        self.fold = args.fold
        self.start_epoch = args.start_epoch
        self.epoches = args.num_epoch
        self.iters = args.num_iters
        self.save_epoch = args.save_epoch

        self.model_name = args.model_name
        self.direction = args.direction

        self.lr_enc = args.lr_enc
        self.lr_seg = args.lr_seg
        self.bs = args.batch_size
        self.n_classes = args.num_classes
        self.srs_rmmax = args.srs_rmmax
        self.tar_rmmax = args.tar_rmmax
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.csco = args.csco
        self.scco = args.scco
        self.threshold = 0.95
        self.best_dice = 0

        A_root = args.A_root
        B_root = args.B_root

        """
        Note that when the dataset is selected as BraTS, 
        the different folds need to be divided based on the patients, and the permutationA and permutationB need to be same.
        """

        if args.permutationA:
            pA = np.load(args.permutationA)
            pA = pA.tolist()
        else:
            pA = list(range(len(os.listdir(A_root))))

        if args.permutationB:
            pB = np.load(args.permutationB)
            pB = pB.tolist()
        else:
            pB = list(range(len(os.listdir(B_root))))

        data_listA = np.array(sorted([os.path.join(A_root, x) for x in os.listdir(A_root)]))[pA].tolist()
        data_listA = K_fold_file_gen(data_listA, self.fold_num, is_shuffle=False)
        train_kA, valid_kA = K_fold_data_gen(data_listA, self.fold - 1, self.fold_num)

        data_listB = np.array(sorted([os.path.join(B_root, x) for x in os.listdir(B_root)]))[pB].tolist()
        data_listB = K_fold_file_gen(data_listB, self.fold_num, is_shuffle=False)
        train_kB, valid_kB = K_fold_data_gen(data_listB, self.fold - 1, self.fold_num)

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        self.checkpoint_dir = os.path.join(args.checkpoint_root,
                                           self.model_name + '_' + timestamp + '_' + str(self.direction))
        crt_file(self.checkpoint_dir)
        self.checkpoint_ = self.checkpoint_dir + "/fold_" + str(args.fold)
        crt_file(self.checkpoint_)

        # Data augmentation
        self.spatial_aug = SpatialTransform(do_rotation=True,
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

        # initialize model
        self.enc = UNet_base(input_channels=1).cuda()
        self.seg = Seg(out=self.n_classes).cuda()

        # initialize teacher model
        self.enc_ema = UNet_base(input_channels=1)
        self.seg_ema = Seg(out=self.n_classes)

        if args.load_pretrain:
            self.load_pretrain(args.pretrain_ckpt_path, args.start_epoch)

        for param in self.enc_ema.parameters():
            param.detach_()
        for param in self.seg_ema.parameters():
            param.detach_()
        self.enc_ema = self.enc_ema.cuda()
        self.seg_ema = self.seg_ema.cuda()

        if args.resume:
            self.load_model(args.load_resume_path, args.start_epoch)

        self.opt_e = torch.optim.Adam(self.enc.parameters(), lr=self.lr_enc)
        self.opt_seg = torch.optim.Adam(self.seg.parameters(), lr=self.lr_seg)

        self.stn = SpatialTransformer()

        if self.direction == "A2B":
            train_srs = train_kA + valid_kA
            train_tar = train_kB
            val_tar = valid_kB
        elif self.direction == "B2A":
            train_srs = train_kB + valid_kB
            train_tar = train_kA
            val_tar = valid_kA

        # initialize the dataloader
        trainsrs_dataset = TrainDataset(train_srs, rmmax=self.srs_rmmax, trans_type=args.transtype)
        traintar_dataset = TrainDataset(train_tar, rmmax=self.tar_rmmax, trans_type=args.transtype)
        test_dataset = TestDataset(val_tar)

        self.dataloader_srstrain = DataLoader(trainsrs_dataset, batch_size=self.bs, shuffle=True)
        self.dataloader_tartrain = DataLoader(traintar_dataset, batch_size=self.bs, shuffle=True)
        self.dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # define loss
        self.L_seg = dice_loss

        # define loss log
        self.L_seg_log = AverageMeter(name='L_Seg')
        self.L_fe_log = AverageMeter(name='L_fe')
        self.L_ent_log = AverageMeter(name='L_ent')

    def to_categorical(self, y, num_classes=None):
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

    def get_pseudo_label(self):
        with torch.no_grad():
            # 用于教师模型预测的CP操作样本
            # 定义：tarimg_cp 表示 目标域 嵌入了 源域图块；
            # 定义：srsimg_cp 表示 源域 嵌入了 目标域图块；
            # 得到的mask以及pseudo_label，尽可能与DDSP的形式一致
            self.img_boxes = context_mask_mul(img_size=self.srs_img.shape[2:], box_num=1, min_mask_ratio=1 / 6,
                                              max_mask_ratio=1 / 3)
            self.srs_img_cp = self.srs_img * (1 - self.img_boxes) + self.tar_img * self.img_boxes
            self.tar_img_cp = self.tar_img * (1 - self.img_boxes) + self.srs_img * self.img_boxes

            # 目标域图像生成伪标签
            prob_ulb_x_w = self.seg_ema(self.enc_ema(self.tar_img))
            prob, self.pseudo_label = torch.max(prob_ulb_x_w, dim=1)  # p^w 与 p^hat，实际是啥等后续调试再看
            # 因为是三维模型，做batch运算则显存肯定不够，干脆去掉batch维度，方便后续处理
            # 当然，这只是权宜之计，后续得改
            # 24.12.5  23:08    验证完毕，实际上不转成此形式也可行！
            # 24.12.6  1:28     没关系了，反正要和DDSP一致，方便操作
            self.pseudo_label = self.pseudo_label[0]
            self.tar_mask = (prob > self.threshold)[0].float()  # 目标域可信度图 w
            self.tar_mask_semantic = (self.pseudo_label & self.tar_mask.bool()).bool()

            # 应用了CP的源域图像生成伪标签，也就是说，中间某个块被CP替换成目标域
            prob_w_lu = self.seg_ema(self.enc_ema(self.srs_img_cp))  # p^w_out，同上
            conf_w_lu, self.pseudo_label_src_cp = torch.max(prob_w_lu, dim=1)  # p^hat_out
            self.pseudo_label_src_cp = self.pseudo_label_src_cp[0]
            self.src_mask_cp = (conf_w_lu > self.threshold)[0].float()  # w_out，它是用于合成mask_w的，同时用于feature计算

            # Fix: Use GT for Source regions in srs_img_cp
            srs_label_indices = torch.argmax(self.srs_label, dim=1)[0]
            box_mask = self.img_boxes.squeeze()
            if box_mask.ndim < srs_label_indices.ndim:
                box_mask = self.img_boxes.view(srs_label_indices.shape)
            
            # srs_img_cp = srs * (1-box) + tar * box. Source is where box==0.
            source_region = (box_mask == 0)
            self.pseudo_label_src_cp[source_region] = srs_label_indices[source_region]
            self.src_mask_cp[source_region] = 1.0

            self.src_mask_cp_semantic = (self.pseudo_label_src_cp & self.src_mask_cp.bool()).bool()

            # 应用了CP的目标域图像生成伪标签，也就是说，中间某个块被CP替换成源头域
            prob_w_ul = self.seg_ema(self.enc_ema(self.tar_img_cp))
            conf_w_ul, self.pseudo_label_tar_cp = torch.max(prob_w_ul, dim=1)  # p^w_in 与 p^hat_in，同上
            self.pseudo_label_tar_cp = self.pseudo_label_tar_cp[0]
            self.tar_mask_cp = (conf_w_ul > self.threshold)[0].float()  # w_in

            # Fix: Use GT for Source regions in tar_img_cp
            # tar_img_cp = tar * (1-box) + srs * box. Source is where box==1.
            source_region_tar = (box_mask == 1)
            self.pseudo_label_tar_cp[source_region_tar] = srs_label_indices[source_region_tar]
            self.tar_mask_cp[source_region_tar] = 1.0

            self.tar_mask_cp_semantic = (self.pseudo_label_tar_cp & self.tar_mask_cp.bool()).bool()

            # 把预测结果再换回来，对应论文的p^hat^mg, mg表示merge
            self.mask_w = self.tar_mask_cp * (1 - self.img_boxes) + self.src_mask_cp * self.img_boxes
            # 生成 w^ens，ens 表示 ensemble
            self.pseudo_label_w = (
                        self.pseudo_label_tar_cp * (1 - self.img_boxes) + self.pseudo_label_src_cp * self.img_boxes)
            ensemble = (self.pseudo_label_w == self.pseudo_label).float() * self.tar_mask
            self.mask_w[ensemble == 0] = 0

            self.pseudo_label = torch.from_numpy(
                self.to_categorical(self.pseudo_label.cpu().numpy(), num_classes=self.n_classes)[np.newaxis, :, :, :,
                :]).cuda()  # 目标域伪标签
            self.pseudo_label_tar_cp = torch.from_numpy(
                self.to_categorical(self.pseudo_label_tar_cp.cpu().numpy(), num_classes=self.n_classes)[np.newaxis, :,
                :, :, :]).cuda()
            self.pseudo_label_src_cp = torch.from_numpy(
                self.to_categorical(self.pseudo_label_src_cp.cpu().numpy(), num_classes=self.n_classes)[np.newaxis, :,
                :, :, :]).cuda()
            self.pseudo_label_w = torch.from_numpy(
                self.to_categorical(self.pseudo_label_w.cpu().numpy(), num_classes=self.n_classes)[np.newaxis, :, :, :,
                :]).cuda()

            # CP增强域图像生成在此处
            # 先进行srs_img 与 tar_img_r，以及 srs_img_r 与 tar_img 的互换
            self.tar_r_mask, self.src_r_mask = self.tar_mask.clone(), self.tar_mask.clone()
            # 增强域图像生成
            self.tar_img_r_cp = self.tar_img_r * (1 - self.img_boxes) + self.srs_img * self.img_boxes
            # 下面两行可能并无必要，因为此时还只是特征层阶段   好吧，有必要，要用mask来计算
            # self.ps_label_tar_r = self.pseudo_label * (1 - self.img_boxes) + self.srs_label * self.img_boxes
            self.tar_r_mask[self.img_boxes.expand(self.tar_r_mask.shape) == 1] = 1

            self.srs_img_r_cp = self.srs_img_r * (1 - self.img_boxes) + self.tar_img * self.img_boxes
            # self.ps_label_src_r = self.srs_label * (1 - self.img_boxes) + self.pseudo_label * self.img_boxes
            self.src_r_mask[self.img_boxes.expand(self.src_r_mask.shape) == 0] = 1

            self.tar_mask = self.tar_mask.to(torch.bool)
            self.tar_r_mask = self.tar_r_mask.to(torch.bool)
            self.src_r_mask = self.src_r_mask.to(torch.bool)
            self.mask_w = self.mask_w.to(torch.bool)
            self.src_mask_cp = self.src_mask_cp.to(torch.bool)
            self.tar_mask_cp = self.tar_mask_cp.to(torch.bool)

    def forward_enc(self):
        self.latent_a = self.enc(self.srs_img)
        self.latent_a_r = self.enc(self.srs_img_r)
        self.latent_a_cp = self.enc(self.srs_img_cp)
        self.latent_a_r_cp = self.enc(self.srs_img_r_cp)
        self.latent_b = self.enc(self.tar_img)
        self.latent_b_r = self.enc(self.tar_img_r)
        self.latent_b_cp = self.enc(self.tar_img_cp)
        self.latent_b_r_cp = self.enc(self.tar_img_r_cp)  # 这一行可能并无必要，但我的新模型应该就有

    def forward_seg(self):  # 改完一半，下一步改这里
        self.latent_a = self.enc(self.srs_img)
        self.latent_a_r = self.enc(self.srs_img_r)
        self.latent_a_cp = self.enc(self.srs_img_cp)
        self.latent_a_r_cp = self.enc(self.srs_img_r_cp)
        self.latent_b = self.enc(self.tar_img)
        self.latent_b_r = self.enc(self.tar_img_r)
        self.latent_b_cp = self.enc(self.tar_img_cp)
        self.latent_b_r_cp = self.enc(self.tar_img_r_cp)

        self.pred_mask_real_a = self.seg(self.latent_a)  #
        self.pred_mask_real_a_cp = self.seg(self.latent_a_cp)
        self.pred_mask_real_a_r = self.seg(self.latent_a_r)  #
        self.pred_mask_real_a_r_cp = self.seg(self.latent_a_r_cp)  #
        self.pred_mask_b = self.seg(self.latent_b)  #
        self.pred_mask_real_b_cp = self.seg(self.latent_b_cp)
        self.pred_mask_b_r = self.seg(self.latent_b_r)  #
        self.pred_mask_real_b_r_cp = self.seg(self.latent_b_r_cp)  #

    def compute_fea_loss(self):

        ###self-sim loss
        C = self.latent_a.shape[1]

        # 生成的特征图要靠近
        # 现在我有伪标签了，我可以对目标域进行类似于下面的相同操作
        # src_cs_loss 表示 原本的SFA损失，对于目标域则应用伪标签
        # src_cp_cs_loss 表示 增强图像的SFA 损失，其计算的是增强图像copy-paste 与 未增强图像 copy-paste 的损失
        # mask使用的是增强图像的mask
        src_cs_loss = crosschannel_sim(self.latent_a[:, :, self.srs_mask].reshape(1, C, -1),
                                       self.latent_a_r[:, :, self.srs_mask].reshape(1, C, -1))

        # 这里有问题，不能直接用可信度图来计算，这与有标签图的mask不符；
        # 而可信度图则将背景也包括在内了，不合理！
        tar_cs_loss = crosschannel_sim(self.latent_b[:, :, self.tar_mask_semantic].reshape(1, C, -1),
                                       self.latent_b_r[:, :, self.tar_mask_semantic].reshape(1, C, -1))

        src_cp_cs_loss = crosschannel_sim(self.latent_a_cp[:, :, self.src_mask_cp_semantic].reshape(1, C, -1),
                                          self.latent_a_r_cp[:, :, self.src_mask_cp_semantic].reshape(1, C, -1))

        tar_cp_cs_loss = crosschannel_sim(self.latent_b_cp[:, :, self.tar_mask_cp_semantic].reshape(1, C, -1),
                                          self.latent_b_r_cp[:, :, self.tar_mask_cp_semantic].reshape(1, C, -1))

        ### When handling BraTS, remove sc_loss.
        # 对于伪标签分割的数据,必须给一个小的权重;因为前期伪标签必然不可靠
        self.fe_loss = self.csco * (src_cs_loss + self.consist_weight * (
                    tar_cs_loss + self.consist_weight * (src_cp_cs_loss + tar_cp_cs_loss)))

    def compute_seg_loss(self):  # 改完一半，下一步改这里，应当套入基于mask的损失计算
        self.loss_ent = prob_entropyloss(self.pred_mask_b)

        ##seg loss
        #  ----------------------------------------有监督的分割损失----------------------------------------  #
        self.sup_seg_loss = self.L_seg(self.pred_mask_real_a, self.srs_label) + \
                            self.L_seg(self.pred_mask_real_a_r, self.srs_label)

        # 利用伪标签的无监督分割损失
        self.unsup_loss_s_cp = self.L_seg(self.pred_mask_b, self.pseudo_label_w, self.mask_w)

        self.unsup_loss_tar_r_cpm = self.L_seg(self.pred_mask_b_r, self.pseudo_label_w,
                                               self.mask_w)  # 或是  self.tar_mask

        #  ----------------------------------------CP图分割----------------------------------------  #
        self.unsup_loss_src_r_cp = self.L_seg(self.pred_mask_real_a_r_cp,
                                              self.pseudo_label_src_cp, self.src_r_mask)

        self.unsup_loss_tar_r_cp = self.L_seg(self.pred_mask_real_b_r_cp,
                                              self.pseudo_label_tar_cp, self.tar_r_mask)

        self.unsup_loss_src_cp = self.L_seg(self.pred_mask_real_a_cp,
                                            self.pseudo_label_src_cp, self.src_mask_cp)

        self.unsup_loss_tar_cp = self.L_seg(self.pred_mask_real_b_cp,
                                            self.pseudo_label_tar_cp, self.tar_mask_cp)

        self.seg_loss = (self.sup_seg_loss +
                         self.consist_weight * (self.unsup_loss_src_cp + self.unsup_loss_tar_cp + self.unsup_loss_s_cp +
                         self.consist_weight * ( self.unsup_loss_src_r_cp + self.unsup_loss_tar_r_cp + self.unsup_loss_tar_r_cpm)))

        # self.seg_loss = (self.sup_seg_loss +
        #                  self.consist_weight * (self.unsup_loss_src_cp + self.unsup_loss_tar_cp + self.unsup_loss_s_cp))

    def train_iterator(self, img1, img1_r, img2, img2_r, img1_label, epoch, iters):
        Depth = img1.shape[2]

        self.srs_img = img1
        self.srs_img_r = img1_r
        self.tar_img = img2
        self.tar_img_r = img2_r

        self.srs_label = img1_label
        self.get_pseudo_label()
        self.consist_weight = get_current_consistency_weight(self.epoch)
        ##Encoder forward
        self.forward_enc()
        
        # Update Prototypes with Source Features
        with torch.no_grad():
            self.proto_bank.update(self.latent_a, self.srs_label)
            
        self.compute_fea_loss()

        self.opt_e.zero_grad()
        self.fe_loss.backward()
        self.opt_e.step()

        ###Seg forward
        self.forward_seg()
        self.compute_seg_loss()

        self.opt_e.zero_grad()
        self.opt_seg.zero_grad()
        self.sen_loss = self.seg_loss + self.loss_ent
        self.sen_loss.backward()

        self.opt_e.step()
        self.opt_seg.step()

        self.L_seg_log.update(self.seg_loss.data, img1.size(0))
        self.L_ent_log.update(self.loss_ent.data, img1.size(0))
        # self.L_consist_log.update(self.target_consist.data, img1.size(0))
        self.L_fe_log.update(self.fe_loss.data, img1.size(0))

    def train_epoch(self, epoch):
        self.enc.train()
        self.seg.train()
        for i in range(self.iters):
            srsimg, srsimg_r, srslabel = next(self.dataloader_srstrain.__iter__())
            tarimg, tarimg_r, _ = next(self.dataloader_tartrain.__iter__())

            srsimg = srsimg.cuda()
            srsimg_r = srsimg_r.cuda()
            tarimg = tarimg.cuda()
            tarimg_r = tarimg_r.cuda()
            srslabel = srslabel.cuda()

            # Augment the source image and target image
            mat, code_spa = self.spatial_aug.rand_coords(srsimg.shape[2:])
            # 下一步操作：对于srsimg 或是 tarimg 进行 strong transform，用于学生模型的训练
            # 原始DDSP除了shuffle remap 之外，可以看成是弱变换 （weak transform）
            # 或者：shuffle remap 本身就可以看成是一种很强大的strong transform
            # 但这样会带来问题：由于shuffle remap 太tmd变态级别的强大了，很可能导致学生模型的崩溃
            # 因此，考虑引入24CVPR的strong，其实际上看作为medium，用于引导学生模型训练；
            # 不过，也可以先尝试使用学生模型分割sr-remap，看看效果
            srsimg = self.spatial_aug.augment_spatial(srsimg, mat, code_spa)
            srslabel = self.spatial_aug.augment_spatial(srslabel, mat, code_spa, mode="nearest").int()
            srsimg_r = self.spatial_aug.augment_spatial(srsimg_r, mat, code_spa)
            tarimg = self.spatial_aug.augment_spatial(tarimg, mat, code_spa)
            tarimg_r = self.spatial_aug.augment_spatial(tarimg_r, mat, code_spa)

            # 这里就导致了，该模型用不了三维的batch  24.12.5  23:10
            # 实际上，这里对应的即为CVPR24 MiDSS 的 lb_mask
            self.srs_mask = (srslabel[0][0] > 0).detach()

            srslabel = srslabel.cpu().numpy()[0][0]
            srslabel = torch.from_numpy(
                self.to_categorical(srslabel, num_classes=self.n_classes)[np.newaxis, :, :, :, :]).cuda()

            self.train_iterator(srsimg, srsimg_r, tarimg, tarimg_r, srslabel, epoch, i)
            update_ema_variables(self.enc, self.enc_ema, 0.99, self.epoch * self.iters + i + 1)
            update_ema_variables(self.seg, self.seg_ema, 0.99, self.epoch * self.iters + i + 1)

            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             self.L_seg_log.__str__(),
                             self.L_ent_log.__str__(),
                             self.L_fe_log.__str__()])
            print(res)

    def checkpoint(self, epoch, is_best=False):
        if is_best:
            torch.save(self.enc.state_dict(), '{0}/ec_epoch_best.pth'.format(self.checkpoint_))
            torch.save(self.seg.state_dict(), '{0}/seg_epoch_best.pth'.format(self.checkpoint_))
            torch.save(self.enc_ema.state_dict(),
                       '{0}/ec_ema_epoch_best.pth'.format(self.checkpoint_))
            torch.save(self.seg_ema.state_dict(),
                       '{0}/seg_ema_epoch_best.pth'.format(self.checkpoint_))
        else:
            torch.save(self.enc.state_dict(), '{0}/ec_epoch_{1}.pth'.format(self.checkpoint_, epoch + self.start_epoch))
            torch.save(self.seg.state_dict(),
                       '{0}/seg_epoch_{1}.pth'.format(self.checkpoint_, epoch + self.start_epoch))
            torch.save(self.enc_ema.state_dict(),
                       '{0}/ec_ema_epoch_{1}.pth'.format(self.checkpoint_, epoch + self.start_epoch))
            torch.save(self.seg_ema.state_dict(),
                       '{0}/seg_ema_epoch_{1}.pth'.format(self.checkpoint_, epoch + self.start_epoch))


    def load_model(self, path, epoch):
        print("loading model epoch ", str(epoch))
        self.enc.load_state_dict(torch.load('{0}/ec_epoch_{1}.pth'.format(path, epoch)), strict=True)
        self.seg.load_state_dict(torch.load('{0}/seg_epoch_{1}.pth'.format(path, epoch)), strict=True)
        self.enc_ema.load_state_dict(torch.load('{0}/ec_ema_epoch_{1}.pth'.format(path, epoch)), strict=True)
        self.seg_ema.load_state_dict(torch.load('{0}/seg_ema_epoch_{1}.pth'.format(path, epoch)), strict=True)

    def load_pretrain(self, path, epoch):
        print("loading pretrained teacher model")
        self.enc_ema.load_state_dict(torch.load('{0}/ec_pretrain.pth'.format(path)), strict=True)
        self.seg_ema.load_state_dict(torch.load('{0}/seg_pretrain.pth'.format(path)), strict=True)

    def train(self):
        self.trainwriter = LogWriter(name=self.checkpoint_ + "/train_" + self.model_name,
                                     head=["epoch", 'loss_ent', 'loss_fe', 'loss_seg', 'loss_all'])

        for epoch in range(self.epoches - self.start_epoch):
            self.L_seg_log.reset()
            self.L_ent_log.reset()
            # self.L_consist_log.reset()
            self.L_fe_log.reset()

            self.epoch = epoch
            self.train_epoch(epoch + self.start_epoch)
            loss_all = self.L_ent_log.avg.item() + self.L_fe_log.avg.item() + self.L_seg_log.avg.item()
            self.trainwriter.writeLog([epoch + self.start_epoch, self.L_ent_log.avg.item(),
                                       self.L_fe_log.avg.item(), self.L_seg_log.avg.item(), loss_all])

            self.test_and_save_best(epoch)

            if epoch % self.save_epoch == 0:
                self.checkpoint(epoch)

        self.checkpoint(self.epoches - self.start_epoch)

    def test_and_save_best(self, epoch):
        self.enc.eval()
        self.seg.eval()

        loss_list = []
        cls = ['AA', 'LAC', 'LVC', 'MYO']
        cls_dice_list = []
        cls_asd_list = []
        len_dataloader = len(self.dataloader_test)
        data_test_iter = iter(self.dataloader_test)

        for i in range(len_dataloader):
            tarimg, tarlabel = data_test_iter.__next__()
            Depth = tarimg.shape[2]

            if torch.cuda.is_available():
                tarimg = tarimg.cuda()
                tarlabel = tarlabel.cuda()

            tarlabel = tarlabel.cpu().numpy()[0][0]
            tarlabel = torch.from_numpy(
                self.to_categorical(tarlabel, num_classes=self.n_classes)[np.newaxis, :, :, :, :]).cuda()

            ###验证损失
            with torch.no_grad():
                latent_b = self.enc(tarimg)
                pred_mask_b = self.seg(latent_b)
                loss_seg = self.L_seg(pred_mask_b, tarlabel)

            tarlab = tarlabel.cpu().numpy()[0]
            tarseg = self.to_categorical(np.argmax(pred_mask_b[0].cpu().numpy(), axis=0), num_classes=self.n_classes)

            tardice_all = []
            tarassd_all = []
            for i in range(self.n_classes - 1):
                tardice_all.append(dice(tarseg[i + 1], tarlab[i + 1]))
                bool_seg = np.where(tarseg[i + 1] == 0, False, True)
                bool_label = np.where(tarlab[i + 1] == 0, False, True)
                surf_dists = surfdist.compute_surface_distances(bool_seg, bool_label, spacing_mm=(1.0, 1.0, 1.0))
                asds = surfdist.compute_average_surface_distance((surf_dists))
                tarassd_all.append((asds[0] + asds[1]) / 2)

            cls_dice_list.append(tardice_all)
            cls_asd_list.append(tarassd_all)
            loss_list.append([loss_seg.item(), np.mean(tardice_all), np.mean(tarassd_all)])

        mean_loss, mean_dice, mean_asd = np.mean(loss_list, 0)

        if mean_dice > self.best_dice:
            self.best_dice = mean_dice
            print('save the best epoch {}'.format(epoch))
            self.checkpoint(epoch, is_best=True)

            cls_dices = np.array(cls_dice_list)
            cls_dices_mean = cls_dices.mean(axis=0)
            cls_dices_deviation = np.std(cls_dices, axis=0)
            cls_asds = np.array(cls_asd_list)
            cls_asds_mean = cls_asds.mean(axis=0)
            cls_asds_deviation = np.std(cls_asds - cls_asds_mean, axis=0)

            cls_losses_text = ''
            for cls, dice_mean, dice_deviation, asd_mean, asd_deviation in zip(cls, cls_dices_mean, cls_dices_deviation,
                                                                               cls_asds_mean, cls_asds_deviation):
                cls_losses_text += (' {} mean dice : {:.2f}%, dice deviation : {:.2f}%, '
                                    'mean asd : {:.2f}, asd deviation : {:.2f}\n').format(
                    cls, dice_mean * 100, dice_deviation * 100,
                    asd_mean, asd_deviation)

            print("Testing loss : {:.2f}".format(mean_loss),
                  "Testing dice : {:.2f}%".format(mean_dice * 100),
                  "Testing asd : {:.2f}".format(mean_asd))
            print(cls_losses_text)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='UDA seg Training Function')

    parser.add_argument('--fold_num', type=int, default=2)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--direction', default="B2A")
    parser.add_argument('--srs_rmmax', type=int, default=10)
    parser.add_argument('--tar_rmmax', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--num_epoch', type=int, default=800)
    parser.add_argument('--num_iters', type=int, default=12)
    parser.add_argument('--save_epoch', type=int, default=50)
    parser.add_argument('--model_name', default="DDSPSeg")
    parser.add_argument("--consistency", type=float, default=1, help="consistency")
    parser.add_argument("--consistency_rampup", type=float, default=500.0, help="consistency_rampup")

    parser.add_argument('--lr_enc', type=int, default=5e-4)
    parser.add_argument('--lr_seg', type=int, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=1)

    # for training in MMWHS dataset
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--A_root', default="./mmwhs_96/ct_debugging96")
    parser.add_argument('--B_root', default="./mmwhs_96/mr_debugging96")

    # for training in Pro12 dataset
    # parser.add_argument('--num_classes', type=int, default=2)
    # parser.add_argument('--A_root', default="./Pro128/BIDMC")
    # parser.add_argument('--B_root', default="./Pro128/HK")

    parser.add_argument('--permutationA', default=None)
    parser.add_argument('--permutationB', default=None)

    parser.add_argument('--transtype', type=str, choices=['Remap', 'BC'], default='Remap')
    parser.add_argument('--csco', type=float, default=0.1)
    parser.add_argument('--scco', type=float, default=0.1)
    parser.add_argument('--checkpoint_root', default="./checkpoint/mmwhs")
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')

    parser.add_argument('--load_pretrain', default=False, help='load pretrain teacher or not')
    parser.add_argument('---pretrain_ckpt_path', default='./checkpoint/pretrain_ct_96', help='load path of pretrained ckpt')
    parser.add_argument('--resume', default=False, help='resume or not')
    parser.add_argument('--load_resume_path', default=None, help='load path of ckpt for resume')

    args = parser.parse_args()

    trainer = DDSPSeg(args=args)
    trainer.train()

