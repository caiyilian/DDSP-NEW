import os
import torch
from torch import nn
from models.network import Seg, UNet_base
from torch.utils.data import DataLoader
from utils.dataloader_origin import Dataset3D as TestDataset
from utils.losses import dice_loss
from utils.utils import AverageMeter, dice, K_fold_data_gen, K_fold_file_gen
import numpy as np
import surface_distance as surfdist

def crt_file(path):
    os.makedirs(path, exist_ok=True)

class DDSPSeg_test(object):
    def __init__(self, args=None):
        super(DDSPSeg_test, self).__init__()

        self.fold_num = args.fold_num
        self.fold = args.fold


        self.direction = args.direction
        self.n_classes = args.num_classes

        A_root = args.A_root
        B_root = args.B_root

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

        self.checkpoint = args.checkpoint
        self.load_epoch = args.load_epoch

        data_listA = np.array(sorted([os.path.join(A_root, x) for x in os.listdir(A_root)]))[pA].tolist()
        data_listA = K_fold_file_gen(data_listA, self.fold_num, is_shuffle=False)
        _, valid_kA = K_fold_data_gen(data_listA, self.fold-1, self.fold_num)

        data_listB = np.array(sorted([os.path.join(B_root, x) for x in os.listdir(B_root)]))[pB].tolist()
        data_listB = K_fold_file_gen(data_listB, self.fold_num, is_shuffle=False)
        _, valid_kB = K_fold_data_gen(data_listB, self.fold-1, self.fold_num)

        # initialize model
        self.enc = UNet_base(input_channels=1).cuda()
        self.seg = Seg(out=self.n_classes).cuda()

        if self.direction == "A2B": 
            test = valid_kB
        elif self.direction == "B2A": 
            test = valid_kA

        # initialize the dataloader
        test_dataset = TestDataset(test)
        self.dataloader_test = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
        # define loss
        self.L_seg = dice_loss

        # define loss log
        self.L_seg_log = AverageMeter(name='L_Seg')

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

    def load_model(self):
        path = self.checkpoint
        epoch = self.load_epoch
        print("loading model epoch ", str(epoch))
        print("checkpoint_path: {}".format(path))
        print("evaluate fold: {}".format(args.fold))
        # self.enc.load_state_dict(torch.load('{0}/ec_epoch_{1}.pth'.format(path, epoch)),strict=True) #, strict=True
        # self.seg.load_state_dict(torch.load('{0}/seg_epoch_{1}.pth'.format(path, epoch)),strict=True)
        self.enc.load_state_dict(torch.load('{0}/ec_epoch_{1}.pth'.format(path, epoch), weights_only=True), strict=True)  # , strict=True
        self.seg.load_state_dict(torch.load('{0}/seg_epoch_{1}.pth'.format(path, epoch), weights_only=True), strict=True)


    def test(self):
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

            # simple_save_visual_pred(pred_mask_b, 'pred_b.nii')

            tarlab = tarlabel.cpu().numpy()[0]
            tarseg = self.to_categorical(np.argmax(pred_mask_b[0].cpu().numpy(), axis=0), num_classes=self.n_classes)

            tardice_all = []
            tarassd_all = []
            for i in range(self.n_classes - 1):
                tardice_all.append(dice(tarseg[i + 1], tarlab[i + 1]))
                bool_seg = np.where(tarseg[i + 1]==0, False, True)
                bool_label = np.where(tarlab[i + 1]==0, False, True)
                surf_dists = surfdist.compute_surface_distances(bool_seg, bool_label, spacing_mm=(1.0, 1.0, 1.0))
                asds = surfdist.compute_average_surface_distance((surf_dists))
                tarassd_all.append((asds[0] + asds[1])/2)

            cls_dice_list.append(tardice_all)
            cls_asd_list.append(tarassd_all)
            loss_list.append([loss_seg.item(), np.mean(tardice_all), np.mean(tarassd_all)])

        mean_loss, mean_dice, mean_asd = np.mean(loss_list, 0)

        cls_dices = np.array(cls_dice_list)
        cls_dices_mean = cls_dices.mean(axis=0)
        cls_dices_deviation = np.std(cls_dices, axis=0)
        cls_asds = np.array(cls_asd_list)
        cls_asds_mean = cls_asds.mean(axis=0)
        cls_asds_deviation = np.std(cls_asds - cls_asds_mean, axis=0)

        cls_losses_text = ''
        for cls, dice_mean, dice_deviation, asd_mean, asd_deviation in zip(cls, cls_dices_mean, cls_dices_deviation, cls_asds_mean, cls_asds_deviation):
            cls_losses_text += (' {} mean dice : {:.2f}%, dice deviation : {:.2f}%, '
                                'mean asd : {:.2f}, asd deviation : {:.2f}\n').format(
                            cls, dice_mean*100, dice_deviation*100,
                                asd_mean, asd_deviation)

        print("Testing loss : {:.2f}".format(mean_loss),
              "Testing dice : {:.2f}%".format(mean_dice*100),
              "Testing asd : {:.2f}".format(mean_asd))
        print(cls_losses_text)


def simple_save_visual_pred(pred, filename='img.nii'):
    import SimpleITK as sitk
    pred = pred.max(1)[1].cpu().numpy().astype(np.float64)
    pred = sitk.GetImageFromArray(pred[0])
    sitk.WriteImage(pred, filename)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='UDA seg Testing Function')

    parser.add_argument('--fold_num', type=int, default=2)
    parser.add_argument('--fold', type=int, default=2)
    parser.add_argument('--direction', default="B2A")
    parser.add_argument('--num_classes', type=int, default=2)
    
    # parser.add_argument('--checkpoint',
    #                     default="./checkpoint/mmwhs96/DDSPSeg_20241216_101639_B2A/fold_2")
    
    # parser.add_argument('--checkpoint',
    #                     default="./checkpoint/mmwhs_cp/DDSPSeg_20241206_171505_A2B/fold_2")
                        
    # parser.add_argument('--checkpoint',
    #                     default="./checkpoint/all_merge_pseudo_pro12/DDSPSeg_20250112_112543_B2A/fold_2")
    parser.add_argument('--checkpoint',
                        default="./checkpoint/all_merge_pseudo_pro12/DDSPSeg_20250112_112543_B2A/fold_2")
    parser.add_argument('--load_epoch', type=str, default='best')

    # parser.add_argument('--A_root', default="./mmwhs_96/ct_debugging96")
    # parser.add_argument('--B_root', default="./mmwhs_96/mr_debugging96")

    parser.add_argument('--A_root', default="./Pro128/BIDMC")
    parser.add_argument('--B_root', default="./Pro128/HK")

    parser.add_argument('--permutationA', default=None)
    parser.add_argument('--permutationB', default=None)

    
    args = parser.parse_args()

    testmodel = DDSPSeg_test(args=args)
    testmodel.load_model()
    testmodel.test()


