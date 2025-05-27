from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import weight_reduce_loss
from mmengine.utils import is_list_of

from mmdet3d.registry import MODELS
from mmdet3d.models.losses.lovasz_loss import lovasz_hinge, lovasz_softmax_flat


def flatten_probs(probs: torch.Tensor, labels: torch.Tensor, ignore_index: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Flatten predictions and labels in the batch. Remove tensors whose labels
    equal to 'ignore_index'.

    Args:
        probs (torch.Tensor): Predictions to be modified.
        labels (torch.Tensor): Labels to be modified.
        ignore_index (int, optional): The label index to be ignored.
            Defaults to None.

    Return:
        tuple(torch.Tensor, torch.Tensor): Modified predictions and labels.
    """
    if probs.dim() != 2:  # for input with P*C
        if probs.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probs.size()
            probs = probs.view(B, 1, H, W)
        if probs.dim() == 5:
            B, C, H, W, D = probs.size()
            probs = probs.view(B, C, H, W * D)

        B, C, H, W = probs.size()
        probs = probs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B*H*W, C=P,C
        labels = labels.view(-1)
    if ignore_index is None:
        return probs, labels
    valid = labels != ignore_index
    vprobs = probs[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobs, vlabels


def lovasz_softmax(
    probs: torch.Tensor,
    labels: torch.Tensor,
    classes: Union[str, List[int]] = "present",
    per_sample: bool = False,
    class_weight: List[float] = None,
    reduction: str = "mean",
    avg_factor: Optional[int] = None,
    ignore_index: int = 255,
) -> torch.Tensor:
    """Multi-class Lovasz-Softmax loss.

    Args:
        probs (torch.Tensor): Class probabilities at each
            prediction (between 0 and 1) with shape [B, C, H, W].
        labels (torch.Tensor): Ground truth labels (between 0 and
            C - 1) with shape [B, H, W].
        classes (Union[str, list[int]]): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Defaults to 'present'.
        per_sample (bool): If per_sample is True, compute the loss per
            sample instead of per batch. Defaults to False.
        class_weight (list[float], optional): The weight for each class.
            Defaults to None.
        reduction (str): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_sample is True. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. This parameter only works when per_sample is True.
            Defaults to None.
        ignore_index (Union[int, None]): The label index to be ignored.
            Defaults to 255.

    Returns:
        torch.Tensor: The calculated loss.
    """

    if per_sample:
        loss = [lovasz_softmax_flat(*flatten_probs(prob.unsqueeze(0), label.unsqueeze(0), ignore_index), classes=classes, class_weight=class_weight) for prob, label in zip(probs, labels)]
        loss = weight_reduce_loss(torch.stack(loss), None, reduction, avg_factor)
    else:
        loss = lovasz_softmax_flat(*flatten_probs(probs, labels, ignore_index), classes=classes, class_weight=class_weight)
    return loss


@MODELS.register_module()
class OccLovaszLoss(nn.Module):
    """LovaszLoss.

    This loss is proposed in `The Lovasz-Softmax loss: A tractable surrogate
    for the optimization of the intersection-over-union measure in neural
    networks <https://arxiv.org/abs/1705.08790>`_.

    Args:
        loss_type (str): Binary or multi-class loss.
            Defaults to 'multi_class'. Options are "binary" and "multi_class".
        classes (Union[str, list[int]]): Classes chosen to calculate loss.
            'all' for all classes, 'present' for classes present in labels, or
            a list of classes to average. Defaults to 'present'.
        per_sample (bool): If per_sample is True, compute the loss per
            sample instead of per batch. Defaults to False.
        reduction (str): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_sample is True. Defaults to 'mean'.
        class_weight ([list[float], optional): Weight of each class.
            Defaults to None.
        loss_weight (float): Weight of the loss. Defaults to 1.0.
    """

    def __init__(
        self,
        loss_type: str = "multi_class",
        classes: Union[str, List[int]] = "present",
        per_sample: bool = False,
        reduction: str = "mean",
        class_weight: Optional[List[float]] = None,
        loss_weight: float = 1.0,
        ignore_index: int = 255,
        **kwargs,
    ):
        super().__init__()
        assert loss_type in (
            "binary",
            "multi_class",
        ), "loss_type should be \
                                                    'binary' or 'multi_class'."

        if loss_type == "binary":
            self.cls_criterion = lovasz_hinge
        else:
            self.cls_criterion = lovasz_softmax
        assert classes in ("all", "present") or is_list_of(classes, int)
        if not per_sample:
            assert (
                reduction == "none"
            ), "reduction should be 'none' when \
                                                        per_sample is False."

        self.classes = classes
        self.per_sample = per_sample
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index

    def forward(self, cls_score: torch.Tensor, label: torch.Tensor, avg_factor: int = None, reduction_override: str = None, **kwargs) -> torch.Tensor:
        """Forward function."""
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        # if multi-class loss, transform logits to probs
        if self.cls_criterion == lovasz_softmax:
            cls_score = F.softmax(cls_score, dim=1)

        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score, label, self.classes, self.per_sample, class_weight=class_weight, reduction=reduction, avg_factor=avg_factor, ignore_index=self.ignore_index, **kwargs
        )
        return loss_cls
