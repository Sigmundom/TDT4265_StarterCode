import math
from xmlrpc.client import Boolean
import torch
import torch.nn as nn
from .anchor_encoder import AnchorEncoder
from torchvision.ops import batched_nms


class RetinaNet(nn.Module):
    def __init__(self, 
            feature_extractor: nn.Module,
            anchors,
            loss_objective,
            num_classes: int,
            improved_weight_init: bool):
        super().__init__()
        """
            Implements the SSD network.
            Backbone outputs a list of features, which are gressed to SSD output with regression/classification heads.
        """

        self.feature_extractor = feature_extractor
        self.loss_func = loss_objective
        self.num_classes = num_classes

        assert (all(out_c == self.feature_extractor.out_channels) for out_c in self.feature_extractor.out_channels), "Expected same number of output channels for all feature maps"
        self.out_ch = self.feature_extractor.out_channels[0]

        assert all(num_boxes == anchors.num_boxes_per_fmap[0] for num_boxes in anchors.num_boxes_per_fmap), "All feature maps must have the same number of aspect ratios!"
        self.n_boxes = anchors.num_boxes_per_fmap[0]

        self.regression_head = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.out_ch, self.n_boxes * 4, kernel_size=3, padding=1),
            )
        self.classification_head = nn.Sequential(
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.out_ch, self.n_boxes * self.num_classes, kernel_size=3, padding=1),
            )


        self.anchor_encoder = AnchorEncoder(anchors)
        # if improved_weight_init:
        #     self._init_weights_improved()
        # else:
        #     self._init_weights()

    def _init_weights_improved(self):
        layers = [*self.regression_head, *self.classification_head]
        for layer in layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.constant_(layer.bias, 0)
                nn.init.normal_(layer.weight, std=0.01)
        
        # p = 0.99
        # background_bias = math.log(p * ((self.num_classes-1)/(1-p)))
        # nn.init.constant_(self.classification_head[-1].bias[:self.n_boxes], background_bias)
        # print(self.classification_head[-1].bias)

    def _init_weights(self):
        # Not sure if this also works for deeper heads, but doesn't help to remove it either
        layers = [*self.regression_head, *self.classification_head]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)
        

    def regress_boxes(self, features):
        locations = []
        confidences = []
        for x in features:
            bbox_delta = self.regression_head(x).view(x.shape[0], 4, -1)
            bbox_conf = self.classification_head(x).view(x.shape[0], self.num_classes, -1)
            locations.append(bbox_delta)
            confidences.append(bbox_conf)
        bbox_delta = torch.cat(locations, 2).contiguous()
        confidences = torch.cat(confidences, 2).contiguous()
        return bbox_delta, confidences

    
    def forward(self, img: torch.Tensor, **kwargs):
        """
            img: shape: NCHW
        """
        if not self.training:
            return self.forward_test(img, **kwargs)
        features = self.feature_extractor(img)
        return self.regress_boxes(features)
    
    def forward_test(self,
            img: torch.Tensor,
            imshape=None,
            nms_iou_threshold=0.5, max_output=200, score_threshold=0.05):
        """
            img: shape: NCHW
            nms_iou_threshold, max_output is only used for inference/evaluation, not for training
        """
        features = self.feature_extractor(img)
        bbox_delta, confs = self.regress_boxes(features)
        boxes_ltrb, confs = self.anchor_encoder.decode_output(bbox_delta, confs)
        predictions = []
        for img_idx in range(boxes_ltrb.shape[0]):
            boxes, categories, scores = filter_predictions(
                boxes_ltrb[img_idx], confs[img_idx],
                nms_iou_threshold, max_output, score_threshold)
            if imshape is not None:
                H, W = imshape
                boxes[:, [0, 2]] *= H
                boxes[:, [1, 3]] *= W
            predictions.append((boxes, categories, scores))
        return predictions

 
def filter_predictions(
        boxes_ltrb: torch.Tensor, confs: torch.Tensor,
        nms_iou_threshold: float, max_output: int, score_threshold: float):
        """
            boxes_ltrb: shape [N, 4]
            confs: shape [N, num_classes]
        """
        assert 0 <= nms_iou_threshold <= 1
        assert max_output > 0
        assert 0 <= score_threshold <= 1
        scores, category = confs.max(dim=1)

        # 1. Remove low confidence boxes / background boxes
        mask = (scores > score_threshold).logical_and(category != 0)
        boxes_ltrb = boxes_ltrb[mask]
        scores = scores[mask]
        category = category[mask]

        # 2. Perform non-maximum-suppression
        keep_idx = batched_nms(boxes_ltrb, scores, category, iou_threshold=nms_iou_threshold)

        # 3. Only keep max_output best boxes (NMS returns indices in sorted order, decreasing w.r.t. scores)
        keep_idx = keep_idx[:max_output]
        return boxes_ltrb[keep_idx], category[keep_idx], scores[keep_idx]