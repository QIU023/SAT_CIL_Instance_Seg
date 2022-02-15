
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..box_utils import match, log_sum_exp, decode, center_size, crop, elemwise_mask_iou, elemwise_box_iou
from random import sample
from data import cfg, mask_type, activation_func


class MultiBoxLoss_dis(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1-10) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1-10)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1-10 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, total_num_classes, to_learn_class, distillation, pos_threshold, neg_threshold, negpos_ratio):
        super(MultiBoxLoss_dis, self).__init__()
        self.total_num_classes = total_num_classes
        self.to_learn_class = to_learn_class
        self.extend = cfg.extend
        self.distillation = distillation
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.negpos_ratio = negpos_ratio

        # If you output a proto mask with this area, your l1 loss will be l1_alpha
        # Note that the area is relative (so 1-10 would be the entire image)
        self.l1_expected_area = 20 * 20 / 70 / 70
        self.l1_alpha = 0.1

        if cfg.use_class_balanced_conf:
            self.class_instances = None
            self.total_instances = 0

    def forward(self,net, sub_predictions,preds, proto,proto_sub,targets, masks, num_crowds):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            mask preds, and prior boxes from SSD net.
                loc shape: torch.size(batch_size,num_priors,4)
                conf shape: torch.size(batch_size,num_priors,total_num_classes)
                masks shape: torch.size(batch_size,num_priors,mask_dim)
                priors shape: torch.size(num_priors,4)
                proto* shape: torch.size(batch_size,mask_h,mask_w,mask_dim)

            targets (list<tensor>): Ground truth boxes and labels for a batch,
                shape: [batch_size][num_objs,5] (last idx is the label).

            masks (list<tensor>): Ground truth masks for each object in each image,
                shape: [batch_size][num_objs,im_height,im_width]

            num_crowds (list<int>): Number of crowd annotations per batch. The crowd
                annotations should be the last num_crowds elements of targets and masks.

            * Only if mask_type == lincomb
        """
        MSE_Loss = torch.nn.MSELoss()
        SmoothL1Loss = torch.nn.SmoothL1Loss()

        loc_data = preds['loc']
        conf_data = preds['conf'][:, :, :len(self.to_learn_class) - cfg.extend]
        mask_data = preds['mask']

        loc_data_sub = sub_predictions['loc']
        conf_data_sub = sub_predictions['conf'][:, :, :len(self.to_learn_class) - cfg.extend]
        mask_data_sub = sub_predictions['mask']

        losses_loc = MSE_Loss(loc_data, loc_data_sub)
        losses_conf = MSE_Loss(conf_data[:, :, 1:], conf_data_sub[:, :, 1:])
        losses_mask = MSE_Loss(mask_data, mask_data_sub)

        losses_proto = MSE_Loss(proto_sub,proto)

        losses = (losses_conf + 15 * losses_mask + losses_loc + 15 * losses_proto)*10
        return losses


# def Unbiased_Knowledge_Distillatino_Loss_for_Instance_Seg_Cls_Conf(conf, conf_old, num_total_classes, num_old_classes):
    