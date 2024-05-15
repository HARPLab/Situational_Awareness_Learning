from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils import functional as F
from segmentation_models_pytorch.base.modules import Activation
import torch
import numpy as np
from sklearn.metrics import recall_score, precision_score, average_precision_score

class object_level_Accuracy(base.Metric):
    __name__ = "object_level_accuracy"

    def __init__(self, threshold=0.5, remove_small_objects = True, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.__name__ = "object_level_accuracy"
        self.remove_small_objects = remove_small_objects

    def forward(self, y_pr_raw, y_gt, y_inst):
        
        # get object_ids 
        # get_mask for each vehicle id
        # get prediction for each object
        # calculate accuracy 
        y_pr = torch.argmax(y_pr_raw[:, :2, ...], dim=1)
        y_gt = torch.argmax(y_gt, dim=1)
        # allowed_inds = y_inst[:, 0, :, :] == 10 or y_inst[:, 0, :, :] == 4 or y_inst[:, 0, :, :] == 23
        ids_tensor = y_inst[:, 1, :, :] + y_inst[:, 2, :, :]*256
        accs = []
        preds = []
        raw_preds = []
        gts = []
        for b in range(ids_tensor.shape[0]):
            ids = ids_tensor[b].unique()
            # print(ids)
            for id in ids:
                inds = ids_tensor[b] == id
                if (self.remove_small_objects == True) and torch.sum(inds) < 10:
                    continue
                y_pr_obj = y_pr[b][inds]
                y_gt_obj = y_gt[b][inds]
                val = torch.mode(y_pr_obj)
                obj_pr = val[0].item()
                obj_pr_ind = val[1].item()
                obj_gt = torch.mode(y_gt_obj)[0].item()
                if obj_gt == 2:
                    continue
                # obj_gt = y_gt_obj[0].item()
                # print(id, obj_pr, obj_gt)
                accs.append(obj_pr == obj_gt)
                preds.append(obj_pr)
                raw_preds.append(y_pr_raw[:, :2, ...][b][obj_pr][inds][obj_pr_ind].item())
                gts.append(obj_gt)
        if len(accs) == 0:
            return 0, preds, gts
        return sum(accs)/len(accs), preds, gts, raw_preds
    
class object_level_Precision(base.Metric):
    __name__ = "object_level_precision"

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.__name__ = "object_level_precision"

    def forward(self, y_pr, y_gt, y_inst):
        
        # get object_ids 
        # get_mask for each vehicle id
        # get prediction for each object
        # calculate accuracy 
        y_pr = torch.argmax(y_pr[:, :2, ...], dim=1)
        y_gt = torch.argmax(y_gt, dim=1)
        # allowed_inds = y_inst[:, 0, :, :] == 10 or y_inst[:, 0, :, :] == 4 or y_inst[:, 0, :, :] == 23
        ids_tensor = y_inst[:, 1, :, :] + y_inst[:, 2, :, :]*256
        accs = []
        preds = []
        gts = []
        for b in range(ids_tensor.shape[0]):
            ids = ids_tensor[b].unique()
            for id in ids:
                inds = ids_tensor[b] == id
                y_pr_obj = y_pr[b][inds]
                y_gt_obj = y_gt[b][inds]
                obj_pr = torch.mode(y_pr_obj)[0].item()
                obj_gt = torch.mode(y_gt_obj)[0].item()
                if obj_gt == 2:
                    continue
                # obj_gt = y_gt_obj[0].item()
                # print(id, obj_pr, obj_gt)
                accs.append(obj_pr == obj_gt)
                preds.append(obj_pr)
                gts.append(obj_gt)
        return precision_score(gts, preds), preds, gts
    
class object_level_Recall(base.Metric):
    __name__ = "object_level_recall"

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.__name__ = "object_level_recall"

    def forward(self, y_pr, y_gt, y_inst):
        
        # get object_ids 
        # get_mask for each vehicle id
        # get prediction for each object
        # calculate accuracy 
        y_pr = torch.argmax(y_pr[:, :2, ...], dim=1)
        y_gt = torch.argmax(y_gt, dim=1)
        # allowed_inds = y_inst[:, 0, :, :] == 10 or y_inst[:, 0, :, :] == 4 or y_inst[:, 0, :, :] == 23
        ids_tensor = y_inst[:, 1, :, :] + y_inst[:, 2, :, :]*256
        accs = []
        preds = []
        gts = []
        for b in range(ids_tensor.shape[0]):
            ids = ids_tensor[b].unique()
            # print(ids)
            for id in ids:
                inds = ids_tensor[b] == id
                y_pr_obj = y_pr[b][inds]
                y_gt_obj = y_gt[b][inds]
                obj_pr = torch.mode(y_pr_obj)[0].item()
                obj_gt = torch.mode(y_gt_obj)[0].item()
                if obj_gt == 2:
                    continue
                # obj_gt = y_gt_obj[0].item()
                # print(id, obj_pr, obj_gt)
                accs.append(obj_pr == obj_gt)
                preds.append(obj_pr)
                gts.append(obj_gt)
        
        return recall_score(gts, preds), preds, gts
    
class object_level_AP(base.Metric):
    __name__ = "object_level_AP"

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.__name__ = "object_level_AP"

    def forward(self, y_pr, y_gt, y_inst):
        
        # get object_ids 
        # get_mask for each vehicle id
        # get prediction for each object
        # calculate accuracy 
        y_pr = torch.argmax(y_pr[:, :2, ...], dim=1)
        y_gt = torch.argmax(y_gt, dim=1)
        # allowed_inds = y_inst[:, 0, :, :] == 10 or y_inst[:, 0, :, :] == 4 or y_inst[:, 0, :, :] == 23
        ids_tensor = y_inst[:, 1, :, :] + y_inst[:, 2, :, :]*256
        accs = []
        preds = []
        gts = []
        for b in range(ids_tensor.shape[0]):
            ids = ids_tensor[b].unique()
            # print(ids)
            for id in ids:
                inds = ids_tensor[b] == id
                y_pr_obj = y_pr[b][inds]
                y_gt_obj = y_gt[b][inds]
                obj_pr = torch.mode(y_pr_obj)[0].item()
                obj_gt = torch.mode(y_gt_obj)[0].item()
                if obj_gt == 2:
                    continue
                # obj_gt = y_gt_obj[0].item()
                # print(id, obj_pr, obj_gt)
                accs.append(obj_pr == obj_gt)
                preds.append(obj_pr)
                gts.append(obj_gt)
        
        return average_precision_score(gts, preds), preds, gts

class IoU(base.Metric):
    __name__ = "iou_score"

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )