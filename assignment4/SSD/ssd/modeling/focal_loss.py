import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, anchors, alpha, gamma):
        super().__init__()
        # print('ARGS')
        # print(anchors, alpha, gamma)
        self.scale_xy = 1.0/anchors.scale_xy
        self.scale_wh = 1.0/anchors.scale_wh

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.anchors = nn.Parameter(anchors(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)
        self.alpha = torch.Tensor(alpha).reshape((1, len(alpha))).cuda()
        self.gamma = gamma
        


    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.anchors[:, :2, :])/self.anchors[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.anchors[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()
    
    def forward(self,
            bbox_delta: torch.FloatTensor, confs: torch.FloatTensor,
            gt_bbox: torch.FloatTensor, gt_labels: torch.LongTensor):
        """
        NA is the number of anchor boxes (by default this is 8732)
            bbox_delta: [batch_size, 4, num_anchors]
            confs: [batch_size, num_classes, num_anchors]
            gt_bbox: [batch_size, num_anchors, 4]
            gt_labels = [batch_size, num_anchors]
        """
        gt_bbox = gt_bbox.transpose(1, 2).contiguous() # reshape to [batch_size, 4, num_anchors]
        
        confs= F.softmax(confs, dim=1)
        confs_log= F.log_softmax(confs, dim=1)
 
        num_classes = confs.shape[1]
        gt_labels_one_hot = F.one_hot(gt_labels, num_classes=num_classes).transpose(1,2)
  
        focal_loss = -self.alpha.T * torch.pow((1-confs), self.gamma) * gt_labels_one_hot * confs_log 
        classification_loss = focal_loss.sum()

        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]
        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]
        regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
        num_pos = gt_locations.shape[0]/4
        total_loss = regression_loss/num_pos + classification_loss/num_pos
        to_log = dict(
            regression_loss=regression_loss/num_pos,
            classification_loss=classification_loss/num_pos,
            total_loss=total_loss
        )
        return total_loss, to_log


if __name__ == '__main__':
    bbox_delta = torch.from_numpy(np.load('bbox_delta.npy')).cuda()
    confs = torch.from_numpy(np.load('confs.npy')).cuda()
    gt_bbox = torch.from_numpy(np.load('gt_bbox.npy')).cuda()
    gt_labels = torch.from_numpy(np.load('gt_labels.npy')).cuda()

    f = FocalLoss([0.01, 0, 1, 2, 3, 4, 5, 6, 7], 1)
    loss = f.forward(bbox_delta, confs, gt_bbox, gt_labels)
    print('Finished')
    print(loss)
