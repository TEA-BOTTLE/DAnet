import sys

sys.path.append('../')
import torch

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
import cPickle
import torch
import argparse
import os
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import my_optim
from utils import AverageMeter
from utils import evaluate
from utils.loader import data_loader
from utils.restore import restore
from utils.localization import get_masks
from utils.vistools import save_im_heatmap_box
from models import *
try:
    import cPickle as pickle
except ImportError:
    import pickle

# default settings

LR = 0.001
EPOCH = 200
DISP_INTERVAL = 50

# indexList = [0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 5, 7, 8, 9, 9, 9, 118, 10, 11, 12, 13, 14, 15, 15, 15, 16, 16, 17, 18, 18, 19, 19, 19, 20, 21, 22, 23, 24, 23, 25, 26, 27, 23, 28, 29, 30, 31, 32, 33, 34, 34, 35, 36, 9, 37, 38, 39, 40, 41, 41, 41, 41, 42, 41, 41, 41, 43, 44, 45, 46, 47, 47, 48, 49, 50, 51, 26, 26, 52, 53, 54, 52, 55, 56, 57, 58, 59, 60, 61, 62, 119, 120, 63, 64, 65, 65, 65, 65, 66, 67, 67, 25, 68, 69, 14, 70, 18, 18, 71, 72, 73, 73, 74, 75, 76, 76, 76, 77, 76, 78, 74, 80, 74, 74, 81, 74, 79, 74, 81, 77, 82, 80, 80, 83, 84, 85, 86, 87, 88, 88, 89, 90, 91, 89, 92, 89, 93, 13, 94, 95, 96, 96, 96, 96, 96, 96, 96, 71, 97, 71, 98, 99, 71, 71, 71, 98, 71, 100, 71, 100, 71, 101, 101, 71, 71, 71, 102, 103, 101, 99, 104, 71, 105, 105, 106, 106, 107, 108, 109, 110, 109, 111, 112, 113, 114, 115, 116, 117, 115, 100]

def get_arguments():
    parser = argparse.ArgumentParser(description='ECCV')
    parser.add_argument("--root_dir", type=str, default='')
    parser.add_argument("--img_dir", type=str, default='../../dataset/CUB_200_2011/images')
    parser.add_argument("--train_list", type=str, default='../../dataset/CUB_200_2011/train.txt')
    parser.add_argument("--cos_alpha", type=float, default=0.2)
    parser.add_argument("--train_list0", type=str, default=None)
    parser.add_argument("--train_list1", type=str, default=None)
    parser.add_argument("--train_list2", type=str, default=None)
    parser.add_argument("--test_list", type=str, default='../../dataset/CUB_200_2011/test.txt')
    parser.add_argument("--test_box", type=str, default='../../dataset/CUB_200_2011/test_boxes.txt')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--dataset", type=str, default='cub')
    parser.add_argument("--num_classes", type=int, default=200)
    parser.add_argument("--arch", type=str, default='vgg_v0')
    parser.add_argument("--threshold", type=str, default='0.03,0.04,0.05,0.1,0.15,0.2,0.25,0.3')
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--decay_points", type=str, default='none')
    parser.add_argument("--epoch", type=int, default=EPOCH)
    parser.add_argument("--tencrop", type=str, default='False')
    parser.add_argument("--onehot", type=str, default='False')
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--disp_interval", type=int, default=DISP_INTERVAL)
    parser.add_argument("--snapshot_dir", type=str, default='')
    parser.add_argument("--resume", type=str, default='True')
    parser.add_argument("--restore_from", type=str, default='../snapshots/vgg4cos9/')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--current_epoch", type=int, default=0)

    return parser.parse_args()


def get_model(args):
    model = eval(args.arch).model(num_classes=args.num_classes,
                                  args=args,
                                  threshold=args.threshold)

    model = torch.nn.DataParallel(model, range(args.num_gpu))
    model.cuda()

    if args.resume == 'True':
        restore(args, model, None)

    return model


def val(args):

    # get model
    files = os.listdir(snapDir)
    for file in files:
        if file[-3:] == 'csv':
            continue
        args.restore_from = os.path.abspath(os.path.join(snapDir, file))

        model = get_model(args)
        model.eval()

        # get data
        _, valcls_loader, valloc_loader = data_loader(args, test_path=True)
        assert len(valcls_loader) == len(valloc_loader), \
            'Error! Different size for two dataset: loc({}), cls({})'.format(len(valloc_loader), len(valcls_loader))

        dataToSave = []

        for dat in tqdm(valcls_loader):
            # parse data
            img_path, img, label_in = dat
            label = label_in

            # forward pass
            img, label = img.cuda(), label.cuda()
            img_var, label_var = Variable(img), Variable(label)
            logits = model(img_var)
            logits0 = F.softmax(logits[-1], dim=1).cpu().data.numpy()
            logits1 = F.softmax(logits[-2], dim=1).cpu().data.numpy()
            logits2 = F.softmax(logits[-3], dim=1).cpu().data.numpy()

            cam_map = F.upsample(model.module.get_cam_maps(), size=(28, 28), mode='bilinear', align_corners=True)
            # cam_map = model.module.get_cam_maps()
            cam_map = cam_map.cpu().data.numpy()
            family_maps = F.upsample(model.module.get_family_maps(), size=(28, 28), mode='bilinear', align_corners=True)
            family_maps = family_maps.cpu().data.numpy()
            order_maps = model.module.get_order_maps()
            order_maps = order_maps.cpu().data.numpy()
            top_maps = get_masks(logits0[0], logits1[0], logits2[0], cam_map, family_maps, order_maps, img_path[0], args.input_size,
                                                 args.crop_size, topk=(1, 5), threshold=1.5, mode='union')

            downsample = nn.Conv2d(3, 3, (4, 4), stride=4).cuda()
            dataToSave.append([downsample(img).view(1, 9408).detach().cpu().numpy(), np.resize(top_maps[0][0], (1, 784)), np.resize(top_maps[0][1], (1, 784)), np.resize(top_maps[0][2], (1, 784)), np.resize(top_maps[0][3], (1, 784)),label.cpu().numpy()])

        f = open(os.path.join('../../../my-IDNNs-master/savedatadebug', file[:-8]+'.txt'), 'wb')
        pickle.dump(dataToSave, f)
        f.close()

if __name__ == '__main__':
    args = get_arguments()
    import json
    global snapDir
    snapDir = args.restore_from
    print 'Running parameters:\n'
    print json.dumps(vars(args), indent=4, separators=(',', ':'))
    if not os.path.exists(args.snapshot_dir):
        os.mkdir(args.snapshot_dir)
    val(args)
