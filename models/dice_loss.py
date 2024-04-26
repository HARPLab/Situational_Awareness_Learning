from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import os.path
import sys
home_folder = os.path.expanduser('~')
sys.path.insert(0, os.path.join(home_folder, 'Situational_Awareness_Learning'))
sys.path.insert(0, './models/')

from segmentation_models_pytorch.losses._functional import soft_dice_score, to_tensor
from segmentation_models_pytorch.losses.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE

__all__ = ["DiceLoss"]


class DiceLoss(_Loss):
    def __init__(
            self,
            mode: str,
            classes: Optional[List[int]] = None,
            log_loss: bool = False,
            from_logits: bool = True,
            smooth: float = 0.0,
            ignore_index: Optional[int] = None,
            eps: float = 1e-7,
            class_weights: Optional[List[int]] = None,
    ):
        """Implementation of Dice loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            class_weights: List of weights for each class. If None, weights are equal to 1.0. Example: [0.2, 0.2, 0.6]
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient (a)
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(DiceLoss, self).__init__()
        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.class_weights = class_weights

        if self.class_weights is not None:
            sum_of_weights = sum(self.class_weights)
            for i in range(0, len(self.class_weights)):
                self.class_weights[i] = self.class_weights[i] / sum_of_weights

        self.class_weights_tensor = None
        self.eps = eps
        self.log_loss = log_loss
        self.ignore_index = ignore_index
        self.__name__ = "DiceLoss"

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)
        if self.class_weights is None:
            self.class_weights = [1 / y_true.size(1)] * y_true.size(1)
        if self.mode == MULTICLASS_MODE:
            pass
            # can't do: assert len(self.class_weights) == len(torch.unique(y_true))
            # because y_true may not have all classes in the batch everytime
        else:
            assert len(self.class_weights) == y_true.size(1)

        if self.class_weights_tensor is None:
            # TODO: Add check if GPU or CPU
            if torch.cuda.is_available():
                if isinstance(self.class_weights, torch.Tensor):
                    self.class_weights_tensor = self.class_weights.clone().detach().cuda()
                else:
                    self.class_weights_tensor = torch.tensor(self.class_weights).cuda()
            else:
                if isinstance(self.class_weights, torch.Tensor):
                    self.class_weights_tensor = self.class_weights.clone().detach().cpu()
                else:
                    self.class_weights_tensor = torch.tensor(self.class_weights).cpu()

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            y_true = y_true.view(bs, 1, -1)
            y_pred = y_pred.view(bs, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == MULTICLASS_MODE:
            y_true = y_true.view(bs, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # H, C, H*W
            else:
                y_true = F.one_hot(y_true, num_classes)  # N,H*W -> N,H*W, C
                y_true = y_true.permute(0, 2, 1)  # N, C, H*W
            assert len(self.class_weights) == y_true.size(1)            

        if self.mode == MULTILABEL_MODE:
            y_true = y_true.view(bs, num_classes, -1)
            y_pred = y_pred.view(bs, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Made by: https://github.com/pytorch/pytorch/issues/1249#issuecomment-339904369
        loss = torch.multiply(loss, self.class_weights_tensor)

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]
        sum_loss = loss.sum()
        return sum_loss

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)