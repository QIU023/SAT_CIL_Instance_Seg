from data import *
from utils.augmentations import SSDAugmentation, BaseTransform
from utils.functions import MovingAverage, SavePath
from utils.logger import Log
from utils import timer
from layers.modules import MultiBoxLoss, MultiBoxLoss_dis, MultiBoxLoss_expert
from yolact import Yolact
from yolact_expert import Yolact_expert

from tqdm import tqdm

# from modules.segformer_offical.mix_transformer import mit_b2

import os
import sys
import time
import math, random
from pathlib import Path
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import datetime

# Oof
import eval as eval_script

from tqdm import tqdm
from torchvision.transforms import functional as Ftrans
from torchvision.transforms import InterpolationMode

import torch.nn.functional as F

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1-10")


parser = argparse.ArgumentParser(
    description='Yolact Training Script')
parser.add_argument('--batch_size', default=8, type=int,
                    help='Batch size for training')
parser.add_argument('--step', default=0, type=int)
parser.add_argument('--task', default='19-1', type=str)
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter. If this is -1-10, the iteration will be'\
                         'determined from the file name.')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                    help='Initial learning rate. Leave as None to read this from the config.')
parser.add_argument('--distillation',  default=True, type=float,
                    help='use the distillation')
parser.add_argument('--load_distillation_net',
default='weights/1-19/yolact_resnet50_pascal_114_120000.pth', type=str,
                    help='use the distillation')
parser.add_argument('--resume',
default='weights/1-19/yolact_resnet50_pascal_114_120000.pth', type=str,
                    help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.')
parser.add_argument('--load_expert_net',
default='weights/20/yolact_resnet50_pascal_943_50000.pth', type=str,
                    help='use the distillation')

parser.add_argument('--save_folder', default='weights/19+1/',
                    help='Directory for saving checkpoint models.')

parser.add_argument('--momentum', default=None, type=float,
                    help='Momentum for SGD. Leave as None to read this from the config.')
parser.add_argument('--decay', '--weight_decay', default=None, type=float,
                    help='Weight decay for SGD. Leave as None to read this from the config.')
parser.add_argument('--gamma', default=None, type=float,
                    help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')

parser.add_argument('--log_folder', default='logs/',
                    help='Directory for saving logs.')
parser.add_argument('--config', default=None,
                    help='The config object to use.')
parser.add_argument('--save_interval', default=10000, type=int,
                    help='The number of iterations between saving the model.')
parser.add_argument('--validation_size', default=5000, type=int,
                    help='The number of images to use for validation.')
parser.add_argument('--validation_epoch', default=1, type=int,
                    help='Output validation information every n iterations. If -1-10, do no validation.')
parser.add_argument('--extend_class', default=1, type=int,
                    help='The number of extend class')
parser.add_argument('--keep_latest', dest='keep_latest', action='store_true',
                    help='Only keep the latest checkpoint instead of each one.')
parser.add_argument('--keep_latest_interval', default=100000, type=int,
                    help='When --keep_latest is on, don\'t delete the latest file at these intervals. This should be a multiple of save_interval or 0.')
parser.add_argument('--dataset', default=None, type=str,
                    help='If specified, override the dataset specified in the config with this one (example: coco2017_dataset).')
parser.add_argument('--no_log', dest='log', action='store_false',
                    help='Don\'t log per iteration information into log_folder.')
parser.add_argument('--log_gpu', dest='log_gpu', action='store_true',
                    help='Include GPU information in the logs. Nvidia-smi tends to be slow, so set this with caution.')
parser.add_argument('--no_interrupt', dest='interrupt', action='store_false',
                    help='Don\'t save an interrupt when KeyboardInterrupt is caught.')
parser.add_argument('--batch_alloc', default=None, type=str,
                    help='If using multiple GPUS, you can set this to be a comma separated list detailing which GPUs should get what local batch size (It should add up to your total batch size).')
parser.add_argument('--no_autoscale', dest='autoscale', action='store_false',
                    help='YOLACT will automatically scale the lr and the number of iterations depending on the batch size. Set this if you want to disable that.')

parser.set_defaults(keep_latest=False, log=False, log_gpu=False, interrupt=True, autoscale=True)
args = parser.parse_args()


if args.config is not None:
    set_cfg(args.config)
    cfg.step = args.step
    args.return_attn = cfg.loss_type == 'SAT_loss'
    cfg.return_attn = args.return_attn

if args.dataset is not None:
    set_dataset(args.dataset)

if args.autoscale and args.batch_size != 8:
    factor = args.batch_size / 8
    if __name__ == '__main__':
        print('Scaling parameters by %.2f to account for a batch size of %d.' % (factor, args.batch_size))

    cfg.lr *= factor
    cfg.max_iter //= factor
    cfg.lr_steps = [x // factor for x in cfg.lr_steps]



# Update training parameters from the config if necessary
def replace(name):
    if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))
replace('lr')
replace('decay')
replace('gamma')
replace('momentum')

# This is managed by set_lr
cur_lr = args.lr

if torch.cuda.device_count() == 0:
    print('No GPUs detected. Exiting...')
    exit(-1)

if args.batch_size // torch.cuda.device_count() < 6:
    if __name__ == '__main__':
        print('Per-GPU batch size is less than the recommended limit for batch norm. Disabling batch norm.')
    cfg.freeze_bn = True

loss_types = ['B', 'C', 'M', 'P', 'D', 'E', 'S', 'I', 'SAT']
fullname = {
    'B': 'Box',
    'C': 'Class',
    'M': 'Mask',
    'P': 'P',
    'D': 'Distill',
    'E': 'Expect',
    'S': 'Student',
    'I': 'I',
    'SAT': "Self-Attention Transfer"
}
# loss_types = ['BoundingBox', 'ClassConfidence', 'Mask', 'P', 'Distillation', 'Expect', 'Student', 'I']

class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
            [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])

        return out

if torch.cuda.is_available():
    # if args.cuda:
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

class NetLoss(nn.Module):
    """
    A wrapper for running the network and computing the loss
    This is so we can more efficiently use DataParallel.
    """

    def __init__(self, net=None,sub_net=None,expert_net=None,
        criterion=None,criterion_dis=None,
        criterion_expert=None,criterion_SAT=None):
        super(NetLoss, self).__init__()

        self.net = net
        self.sub_net = sub_net
        self.criterion = criterion
        self.expert = expert_net
        self.criterion_dis = criterion_dis
        self.criterion_expert = criterion_expert
        self.criterion_SAT = criterion_SAT
        self.SAT_weight = 20

    def forward(self, images, targets, masks, num_crowds):
        # print(type(self.net(images, sub=False)))
        # for item in self.net(images, sub=False):
        #     print(item.shape)
        preds,preds_extend,proto, selfattention = self.net(images,sub=False)
        # (preds,preds_extend,proto) = detect
        preds_sub,proto_sub, selfattention_sub = self.sub_net(images,sub=True)
        # (preds_sub,proto_sub) = sub_detect

        losses = self.criterion(self.net, preds_extend, targets, masks, num_crowds)

        if self.criterion_dis is not None:
            losses_dis = self.criterion_dis(self.net, preds_sub, preds,proto,proto_sub, targets, masks, num_crowds)
            losses['D'] = losses_dis
        
        if self.expert is not None:
            preds_expert, proto_expert = self.expert(images, sub=False)
            losses_expert = self.criterion_expert(self.net,preds_expert,preds_extend,proto,proto_expert,targets,masks,num_crowds)
            losses['E'] = losses_expert

        if self.criterion_SAT is not None:
            losses_SAT = self.SAT_weight*self.criterion_SAT(selfattention_sub, selfattention, masks, None)
            losses['SAT'] = losses_SAT

        return losses

class Self_Attention_Transfer_InstanceSeg_Loss(nn.Module):
    def __init__(self, instance_mask_region=True):
        super(Self_Attention_Transfer_InstanceSeg_Loss, self).__init__()

        self.instance_mask_region = instance_mask_region
        self.scale_MiT = [128,64,32,16]
        self.reduction_MiT = [8,4,2,1]
        self.heads_MiT = [2,3,5,8]

    def forward(self, old_network_SelfAttention_arr, new_network_SelfAttention_arr, instance_mask, bbox):
        scale_SA_loss = 0.
        # loss = 0.

        for ii in range(4):
            old_network_SelfAttention = old_network_SelfAttention_arr[ii]
            new_network_SelfAttention = new_network_SelfAttention_arr[ii]
            assert old_network_SelfAttention.shape == new_network_SelfAttention.shape, \
                f"old network:{old_network_SelfAttention.shape} and new network:{new_network_SelfAttention.shape} have different shape Self-Attention Tensor!!!!"
            
#                     resized_labels.append(sc_arr)
            # resize_label = torch.stack(resize_label, dim=0).cuda()
            # print(instance_mask.shape)
            # _, H, W = instance_mask[0].shape
            # assert L == H * W, "input shape  "
            Bs, Heads, L, L_1 = old_network_SelfAttention.shape
            sc = self.scale_MiT[ii]
            
            old_network_SelfAttention = old_network_SelfAttention.permute(0, 2, 1, 3)
            old_network_SelfAttention = old_network_SelfAttention.view(Bs, sc, sc, Heads, L_1)
            
            new_network_SelfAttention = new_network_SelfAttention.permute(0, 2, 1, 3)
            new_network_SelfAttention = new_network_SelfAttention.view(Bs, sc, sc, Heads, L_1)

            # loss = 0.
            if self.instance_mask_region:           # using instance mask to do Region Pooling!
                # resize_label = []
                batch_SA_loss = 0.
                for j in range(Bs):
                    resize_instance_mask_j = []
                    instance_num = instance_mask[j].shape[0]

                    instance_img_mask = instance_mask[j]
                    old_img_SelfAttention = old_network_SelfAttention[j]
                    new_img_SelfAttention = new_network_SelfAttention[j]
                    
                    img_SA_loss = 0.
                    for jj in range(instance_num):
                    # print(instance_mask[j].max(), instance_mask[j].min())
                        lbl_jj = Ftrans.to_pil_image(instance_mask[j][jj].cpu().numpy().astype(np.uint8))
                        lbl_jj = Ftrans.resize(lbl_jj, (sc,sc), InterpolationMode.NEAREST)
                        instance = torch.from_numpy(np.array(lbl_jj)).bool()
                    #     resize_instance_mask_j.append(lbl_jj)
                    # resize_instance_mask_j = torch.stack(resize_instance_mask_j, dim=0)
                    # resize_label.append(resize_instance_mask_j)
                    # batch_size = instance_mask.shape[0]
                    # for i, instance_img_mask in enumerate(resize_label):

                    # for instance in instance_img_mask:
                        old_mean_SelfAttention = old_img_SelfAttention[instance].mean(dim=0)
                        new_mean_SelfAttention = new_img_SelfAttention[instance].mean(dim=0)

                        ins_SA_loss = 0.
                        for hs in range(Heads):
                            old_hs_SA = old_mean_SelfAttention[hs]
                            new_hs_SA = new_mean_SelfAttention[hs]
                            # print(old_hs_SA.shape)
                            
                            old_hs_SA = F.normalize(old_hs_SA, p=2, dim=0)
                            new_hs_SA = F.normalize(new_hs_SA, p=2, dim=0)

                            hs_SA_loss = torch.frobenius_norm(old_hs_SA-new_hs_SA)
                            ins_SA_loss += hs_SA_loss
                        ins_SA_loss /= Heads
                        img_SA_loss += ins_SA_loss
                    img_SA_loss /= instance_img_mask.shape[0]
                    batch_SA_loss += img_SA_loss
                batch_SA_loss /= Bs
                scale_SA_loss += batch_SA_loss
            else:
                pass
            
        scale_SA_loss /= 4
        return scale_SA_loss


class CustomDataParallel(nn.DataParallel):
    """
    This is a custom version of DataParallel that works better with our training data.
    It should also be faster than the general case.
    """

    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in device_ids]
        splits = prepare_data(inputs[0], devices, allocation=args.batch_alloc)

        return [[split[device_idx] for split in splits] for device_idx in range(len(devices))], \
            [kwargs] * len(devices)

    def gather(self, outputs, output_device):
        out = {}

        for k in outputs[0]:
            out[k] = torch.stack([output[k].to(output_device) for output in outputs])

        return out


# def split_classes(cfg):
#     first_num_classes = cfg.first_num_classes
#     learn_num_per_step = int(cfg.task.split('-')[1])
#     for i in range(cfg.step):
#         first_num_classes += learn_num_per_step

#     total_number = cfg.total_num_classes - 1
#     # to_learn
#     original = list(range(total_number + 1))
#     learned_class = []
#     if 'expert' not in cfg.name:
#         learned_class = list(range(first_num_classes+1))
#     current_learn_class = list(range(first_num_classes+1, 1+first_num_classes+learn_num_per_step))
#     remaining = list(range(current_learn_class[-1]+1, total_number+1))
    
#     print(f'learning class: {current_learn_class}, previous learned class: {learned_class}, remain: {remaining} not learned!')

#     return current_learn_class, learned_class, remaining


def split_classes(cfg):
    first_num_classes = cfg.first_num_classes
    if cfg.extend != 0:
        first_num_classes += cfg.extend
    # FIXME loader!

    total_number = 20

    original = list(range(total_number + 1))
    to_learn = list(range(first_num_classes + 1))
    remaining = [i for i in original if i not in to_learn]
    if cfg.extend != 0:
        prefetch_cats = cfg.extend
        prefetch_cats = to_learn[-prefetch_cats:]
    else:
        prefetch_cats = to_learn
    print(f'total learned or learning class: {to_learn}, \n incremental class: {prefetch_cats}, \n remain: {remaining} not learned!')
    return to_learn, prefetch_cats, remaining

def train():
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
  #  CUDA_VISIBLE_DEVICES = [0]
    to_learn, prefetch_classes, remain = split_classes(cfg)
    dataset = COCODetection(image_path=cfg.dataset.train_images,
                            info_file=cfg.dataset.train_info,
                            transform=SSDAugmentation(MEANS),cfg=cfg)
    
    if args.validation_epoch > 0:
        setup_eval()
        val_dataset = COCODetection_test(image_path=cfg.dataset.valid_images,
                                    prefetch_classes=prefetch_classes,
                                    info_file=cfg.dataset.valid_info,
                                    transform=BaseTransform(MEANS))
    #
    sub_net = None
    if cfg.distillation:
        yolact_sub_net = Yolact(sub=True)
        if args.load_distillation_net is not None:
            print('loading distillation net, loading {}...'.format(args.load_distillation_net))
            yolact_sub_net.load_weights(args.load_distillation_net)
            for p in yolact_sub_net.parameters():
                p.requires_grad = False
        sub_net = yolact_sub_net

    expert_net = None
    if cfg.expert:
        yolact_expert_net = Yolact_expert(sub=False)
        if args.load_expert_net is not None:
            print('loading expert net, loading {}...'.format(args.load_expert_net))
            yolact_expert_net.load_weights(args.load_expert_net)
            for p in yolact_expert_net.parameters():
                p.requires_grad = False
        expert_net = yolact_expert_net

    # Parallel wraps the underlying module, but when saving and loading we don't want that
    yolact_net = Yolact(sub=False)
  #  yolact_net.init_weights(backbone_path=args.save_folder + cfg.backbone.path)
    net = yolact_net

    net.train()
   # sub_net.eval()

    if args.log:
        log = Log(cfg.name, args.log_folder, dict(args._get_kwargs()),
            overwrite=(args.resume is None), log_gpu_stats=args.log_gpu)

    # I don't use the timer during training (I use a different timing method).
    # Apparently there's a race condition with multiple GPUs, so disable it just to be safe.
    timer.disable_all()

    # target_id = len([item for item in os.listdir(args.save_folder) if 'interrupt' in item])

    # args.resume = None
    # Both of these can set args.resume to None, so do them before the check  
    # 
    # 
    # print('please input the resume file id:')
    # target_id = int(input())

    # if args.resume == 'interrupt':
    #     args.resume = SavePath.get_interrupt(args.save_folder, target_id)

    # elif args.resume == 'latest':
    #     args.resume = SavePath.get_latest(args.save_folder, cfg.name)

    args.resume = None

    if args.resume is not None:
        print('Initializing weights firstly...')
        pretrain_path = 'weights/mit_b2.pth'
        yolact_net.init_weights(backbone_path=pretrain_path)
        print('Resuming training, loading {}...'.format(args.resume))
        yolact_net.load_weights(args.resume)

        # print('Resuming training,loading expert, loading {}...'.format(args.load_expert_net))
        # yolact_net.load_weights_expert(args.load_expert_net)

        # if args.start_iter == -1:  
            # begin_iter = 
        args.start_iter = int(args.resume[:-4].split('_')[-1])
        print('resume iteration index:', args.start_iter)
        assert args.start_iter > 0
        # raise RuntimeError

    else:
        print('Initializing weights...')
        pretrain_path = 'weights/mit_b2.pth'
        yolact_net.init_weights(backbone_path=pretrain_path)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.decay)
    criterion = MultiBoxLoss(total_num_classes=cfg.total_num_classes,
                             to_learn_class= to_learn,
                             distillation = args.distillation,
                             pos_threshold=cfg.positive_iou_threshold,
                             neg_threshold=cfg.negative_iou_threshold,
                             negpos_ratio=cfg.ohem_negpos_ratio)

    criterion_dis = None
    criterion_expert = None
    criterion_SAT = None
    if cfg.loss_type != 'SAT_loss':
        criterion_dis = MultiBoxLoss_dis(total_num_classes=cfg.total_num_classes,
                                to_learn_class=to_learn,
                                distillation=args.distillation,
                                pos_threshold=cfg.positive_iou_threshold,
                                neg_threshold=cfg.negative_iou_threshold,
                                negpos_ratio=cfg.ohem_negpos_ratio)

        criterion_expert = MultiBoxLoss_expert(total_num_classes=cfg.total_num_classes,
                                to_learn_class=to_learn,
                                distillation=args.distillation,
                                pos_threshold=cfg.positive_iou_threshold,
                                neg_threshold=cfg.negative_iou_threshold,
                                negpos_ratio=cfg.ohem_negpos_ratio)

    # if cfg.loss_type == 'SAT_loss':
    else:
        criterion_SAT = Self_Attention_Transfer_InstanceSeg_Loss(True)


    if args.batch_alloc is not None:
        args.batch_alloc = [int(x) for x in args.batch_alloc.split(',')]
        if sum(args.batch_alloc) != args.batch_size:
            print('Error: Batch allocation (%s) does not sum to batch size (%s).' % (args.batch_alloc, args.batch_size))
            exit(-1)

    net = NetLoss(net, sub_net, expert_net, criterion, criterion_dis,criterion_expert,criterion_SAT)
    # net = CustomDataParallel(NetLoss(net,sub_net, criterion,criterion_dis))
    # if torch.cuda.device_count() > 1:
    net = CustomDataParallel(net)

   # net = NetLoss(net, criterion)
    if args.cuda:
        net = net.cuda()
    
    # Initialize everything
    if not cfg.freeze_bn: yolact_net.freeze_bn() # Freeze bn so we don't kill our means
    yolact_net(torch.zeros(1, 3, cfg.max_size, cfg.max_size).cuda(),sub=False)
    if not cfg.freeze_bn: yolact_net.freeze_bn(True)

    # loss counters

    iteration = max(args.start_iter, 0)
    last_time = time.time()

    epoch_size = len(dataset) // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)
    
    # Which learning rate adjustment step are we on? lr' = lr * gamma ^ step_index
    step_index = 0

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    
    
    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_folder)
    time_avg = MovingAverage()

    global loss_types # Forms the print order
    loss_avgs  = { k: MovingAverage(100) for k in loss_types }

    print('Begin training!')
    print()
    # try-except so you can use ctrl+c to save early and stop training
    try:
        for epoch in range(num_epochs):
            # Resume from start_iter
            if (epoch+1)*epoch_size < iteration:
                continue
            
            tbar = tqdm(data_loader)
            for datum in tbar:
                # Stop if we've reached an epoch if we're resuming from start_iter
                if iteration == (epoch+1)*epoch_size:
                    break

                # Stop at the configured number of iterations even if mid-epoch
                if iteration == cfg.max_iter:
                    break

                # Change a config setting if we've reached the specified iteration
                changed = False
                for change in cfg.delayed_settings:
                    if iteration >= change[0]:
                        changed = True
                        cfg.replace(change[1])

                        # Reset the loss averages because things might have changed
                        for avg in loss_avgs:
                            avg.reset()
                
                # If a config setting was changed, remove it from the list so we don't keep checking
                if changed:
                    cfg.delayed_settings = [x for x in cfg.delayed_settings if x[0] > iteration]

                # Warm up by linearly interpolating the learning rate from some smaller value
                if cfg.lr_warmup_until > 0 and iteration <= cfg.lr_warmup_until:
                    set_lr(optimizer, (args.lr - cfg.lr_warmup_init) * (iteration / cfg.lr_warmup_until) + cfg.lr_warmup_init)

                # Adjust the learning rate at the given iterations, but also if we resume from past that iteration
                while step_index < len(cfg.lr_steps) and iteration >= cfg.lr_steps[step_index]:
                    step_index += 1
                    set_lr(optimizer, args.lr * (args.gamma ** step_index))
                
                # Zero the grad to get ready to compute gradients
                optimizer.zero_grad()

                # Forward Pass + Compute loss at the same time (see CustomDataParallel and NetLoss)
                losses = net(datum)
                
                losses = { k: (v).mean() for k,v in losses.items() } # Mean here because Dataparallel
                loss = sum([losses[k] for k in losses])
                
                # no_inf_mean removes some components from the loss, so make sure to backward through all of it
                # all_loss = sum([v.mean() for v in losses.values()])

                # Backprop

                loss.backward() # Do this to free up vram even if loss is not finite
                if torch.isfinite(loss).item():
                    optimizer.step()
                
                # Add the loss to the moving average for bookkeeping
                for k in losses:
                    loss_avgs[k].add(losses[k].item())

                cur_time  = time.time()
                elapsed   = cur_time - last_time
                last_time = cur_time

                # Exclude graph setup from the timing information
                if iteration != args.start_iter:
                    time_avg.add(elapsed)

                # if iteration % 10 == 0:
                eta_str = str(datetime.timedelta(seconds=(cfg.max_iter-iteration) * time_avg.get_avg())).split('.')[0]
                
                total = sum([loss_avgs[k].get_avg() for k in losses])
                loss_labels = sum([[fullname[k], loss_avgs[k].get_avg()] for k in loss_types if k in losses], [])
                
                tbar.set_description(('[%3d] %7d ||' + (' %s: %.3f |' * len(losses)) + ' T: %.3f || ETA: %s || timer: %.3f || lr:%.3f')
                        % tuple([epoch, iteration] + loss_labels + [total, eta_str, elapsed] + [cur_lr]))

                if args.log:
                    precision = 5
                    loss_info = {k: round(losses[k].item(), precision) for k in losses}
                    loss_info['T'] = round(loss.item(), precision)

                    if args.log_gpu:
                        log.log_gpu_stats = (iteration % 10 == 0) # nvidia-smi is sloooow
                        
                    log.log('train', loss=loss_info, epoch=epoch, iter=iteration,
                        lr=round(cur_lr, 10), elapsed=elapsed)

                    log.log_gpu_stats = args.log_gpu
                
                iteration += 1

                if iteration % args.save_interval == 0 and iteration != args.start_iter:
                    if args.keep_latest:
                        latest = SavePath.get_latest(args.save_folder, cfg.name)

                    print('Saving state, iter:', iteration)
                    yolact_net.save_weights(save_path(epoch, iteration))

                    if args.keep_latest and latest is not None:
                        if args.keep_latest_interval <= 0 or iteration % args.keep_latest_interval != args.save_interval:
                            print('Deleting old save...')
                            os.remove(latest)
            
            # This is done per epoch
            if args.validation_epoch > 0:
                if epoch % args.validation_epoch == 0 and epoch > 0:
                    compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
        
        # Compute validation mAP after training is finished
        compute_validation_map(epoch, iteration, yolact_net, val_dataset, log if args.log else None)
    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')
            
            # Delete previous copy of the interrupted network so we don't spam the weights folder
            # SavePath.remove_interrupt(args.save_folder)
            ckpt_num = len(os.listdir(args.save_folder))
            yolact_net.save_weights(save_path(epoch, repr(iteration) + '_interrupt_'+str(ckpt_num)))
        exit()

    # yolact_net.save_weights(save_path(epoch, iteration))


def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    
    global cur_lr
    cur_lr = new_lr

def gradinator(x):
    x.requires_grad = False
    return x

def prepare_data(datum, devices:list=None, allocation:list=None):
    with torch.no_grad():
        if devices is None:
            devices = ['cuda:0'] if args.cuda else ['cpu']
        if allocation is None:
            allocation = [args.batch_size // len(devices)] * (len(devices) - 1)
            allocation.append(args.batch_size - sum(allocation)) # The rest might need more/less
        
        images, (targets, masks, num_crowds) = datum

        cur_idx = 0
        for device, alloc in zip(devices, allocation):
            for _ in range(alloc):
                images[cur_idx]  = gradinator(images[cur_idx].to(device))
                targets[cur_idx] = gradinator(targets[cur_idx].to(device))
                masks[cur_idx]   = gradinator(masks[cur_idx].to(device))
                cur_idx += 1

        if cfg.preserve_aspect_ratio:
            # Choose a random size from the batch
            _, h, w = images[random.randint(0, len(images)-1)].size()

            for idx, (image, target, mask, num_crowd) in enumerate(zip(images, targets, masks, num_crowds)):
                images[idx], targets[idx], masks[idx], num_crowds[idx] \
                    = enforce_size(image, target, mask, num_crowd, w, h)
        
        cur_idx = 0
        split_images, split_targets, split_masks, split_numcrowds \
            = [[None for alloc in allocation] for _ in range(4)]

        for device_idx, alloc in enumerate(allocation):
            split_images[device_idx]    = torch.stack(images[cur_idx:cur_idx+alloc], dim=0)
            split_targets[device_idx]   = targets[cur_idx:cur_idx+alloc]
            split_masks[device_idx]     = masks[cur_idx:cur_idx+alloc]
            split_numcrowds[device_idx] = num_crowds[cur_idx:cur_idx+alloc]

            cur_idx += alloc

        return split_images, split_targets, split_masks, split_numcrowds

def no_inf_mean(x:torch.Tensor):
    """
    Computes the mean of a vector, throwing out all inf values.
    If there are no non-inf values, this will return inf (i.e., just the normal mean).
    """

    no_inf = [a for a in x if torch.isfinite(a)]

    if len(no_inf) > 0:
        return sum(no_inf) / len(no_inf)
    else:
        return x.mean()

def compute_validation_loss(net, data_loader, criterion):
    global loss_types

    with torch.no_grad():
        losses = {}
        
        # Don't switch to eval mode because we want to get losses
        iterations = 0
        for datum in tqdm(data_loader):
            images, targets, masks, num_crowds = prepare_data(datum)
            out = net(images)

            wrapper = ScatterWrapper(targets, masks, num_crowds)
            _losses = criterion(out, wrapper, wrapper.make_mask())
            
            for k, v in _losses.items():
                v = v.mean().item()
                if k in losses:
                    losses[k] += v
                else:
                    losses[k] = v

            iterations += 1
            if args.validation_size <= iterations * args.batch_size:
                break
        
        for k in losses:
            losses[k] /= iterations
            
        
        loss_labels = sum([[fullname[k], losses[k]] for k in loss_types if k in losses], [])
        # for i in 
        print(('Validation ||' + (' %s: %.3f |' * len(losses)) + ')') % tuple(loss_labels), flush=True)

def compute_validation_map(epoch, iteration, yolact_net, dataset, log:Log=None):
    with torch.no_grad():
        yolact_net.eval()
        
        start = time.time()
        print()
        print("Computing validation mAP (this may take a while)...", flush=True)
        val_info = eval_script.evaluate(yolact_net, dataset, train_mode=True)
        end = time.time()

        if log is not None:
            log.log('val', val_info, elapsed=(end - start), epoch=epoch, iter=iteration)

        yolact_net.train()

def setup_eval():
    eval_script.parse_args(['--no_bar', '--max_images='+str(args.validation_size)])

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  #  torch.cuda.set_device('1')
    train()
