import numpy as np
import torch
from typing import Dict, Optional, Sequence
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from mmdet3d.registry import METRICS
from mmengine.logging import print_log
from terminaltables import AsciiTable


class SSCompute:
    """Computes various evaluation metrics for semantic segmentation."""

    def __init__(self, n_classes: int, free_index: int = 0, ignore_index: int = 255):
        self.n_classes = n_classes
        self.free_index = free_index
        self.ignore_index = ignore_index
        self.reset()

    def add_batch(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        """Add a batch of predictions and ground truth for metric computation."""

        self.tps = self.tps.to(device=y_pred.device)
        self.fps = self.fps.to(device=y_pred.device)
        self.fns = self.fns.to(device=y_pred.device)

        mask = y_true != self.ignore_index
        tp, fp, fn = self._get_score_completion(y_pred, y_true, mask)

        # Accumulate metrics
        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn

        tp_sum, fp_sum, fn_sum = self._get_score_semantic_and_completion(
            y_pred, y_true, mask
        )
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum

    def get_stats(self) -> Dict[str, float]:
        """Return computed statistics (precision, recall, IOU)."""
        if self.completion_tp != 0:
            precision = self.completion_tp / (self.completion_tp + self.completion_fp)
            recall = self.completion_tp / (self.completion_tp + self.completion_fn)
            iou = self.completion_tp / (
                self.completion_tp + self.completion_fp + self.completion_fn
            )
        else:
            precision, recall, iou = 0, 0, 0

        iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)
        return {
            "precision": precision.item(),
            "recall": recall.item(),
            "iou": iou.item(),
            "iou_ssc": [val.item() for val in iou_ssc],
            "iou_ssc_mean": torch.mean(iou_ssc[1:]),  # Excluding background class
        }

    def reset(self) -> None:
        """Reset the metric counters."""
        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0
        self.tps = torch.zeros(self.n_classes, dtype=torch.int64)
        self.fps = torch.zeros(self.n_classes, dtype=torch.int64)
        self.fns = torch.zeros(self.n_classes, dtype=torch.int64)

    def _get_score_completion(
        self, predict: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> tuple:
        """Calculate true positives (TP), false positives (FP), and false negatives (FN) for completion."""
        # Apply mask and compute binary prediction vs target comparison

        predict[~mask] = 0
        target[~mask] = 0

        target_flat = target.reshape(-1)
        predict_flat = predict.reshape(-1)

        b_pred = (predict_flat != self.free_index).to(torch.int64)
        b_true = (target_flat != self.free_index).to(torch.int64)

        tp = torch.sum((b_true == 1) & (b_pred == 1))
        fp = torch.sum((b_true == 0) & (b_pred == 1))
        fn = torch.sum((b_true == 1) & (b_pred == 0))

        return tp, fp, fn

    def _get_score_semantic_and_completion(
        self, predict: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> tuple:
        """Compute TP, FP, FN for semantic segmentation and completion."""
        predict *= mask
        target *= mask

        target_flat = target.reshape(-1)
        predict_flat = predict.reshape(-1)

        # Create class masks and compute TP, FP, FN for each class
        n_class = torch.arange(self.n_classes).to(predict.device)
        class_masks_true = torch.eq(target_flat[:, None], n_class)
        class_masks_pred = torch.eq(predict_flat[:, None], n_class)

        tp_sum = torch.sum(class_masks_true & class_masks_pred, dim=0).to(
            dtype=torch.int64, device=target.device
        )
        fp_sum = torch.sum(~class_masks_true & class_masks_pred, dim=0).to(
            dtype=torch.int64, device=target.device
        )
        fn_sum = torch.sum(class_masks_true & ~class_masks_pred, dim=0).to(
            dtype=torch.int64, device=target.device
        )

        return tp_sum, fp_sum, fn_sum


class BaseSscMetric(BaseMetric):
    """Base class for 3D semantic segmentation metrics."""

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        num_classes: int = 20,
        free_index: int = 0,
        ignore_index: int = 255,
        **kwargs,
    ):
        super().__init__(prefix=prefix, collect_device=collect_device)
        self.time_cost = []
        self.ssc_compute = SSCompute(num_classes, free_index, ignore_index)
        self.ignore_index = ignore_index

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process each batch of predictions and ground truth."""
        for data_sample in data_samples:
            self.ssc_compute.add_batch(data_sample["y_pred"], data_sample["y_true"])
            self.time_cost.append(data_sample["time_cost"])

    def log_show(
        self, stats: Dict[str, float], label2cat: dict, ignore_index: int, logger=None
    ) -> Dict[str, float]:
        """Display metrics in a table and return them as a dictionary."""
        precision = stats["precision"]
        recall = stats["recall"]
        iou = stats["iou"]
        iou_ssc = stats["iou_ssc"]
        miou = stats["iou_ssc_mean"]
        time_cost = np.mean(self.time_cost)
        fps = 1 / time_cost

        header = (
            ["classes"]
            + [label2cat[i] for i in range(len(label2cat))]
            + ["iou", "miou", "precision", "recall", "time", "FPS"]
        )
        table_columns = (
            [["results"]]
            + [[f"{iou_ssc[i]:.4f}"] for i in range(len(label2cat))]
            + [
                [f"{iou:.4f}"],
                [f"{miou:.4f}"],
                [f"{precision:.4f}"],
                [f"{recall:.4f}"],
                [f"{time_cost:.4f}"],
                [f"{fps:.4f}"],
            ]
        )

        ret_dict = {label2cat[i]: float(iou_ssc[i]) for i in range(len(label2cat))}
        ret_dict.update(
            {
                "iou": float(iou),
                "miou": float(miou),
                "precision": float(precision),
                "recall": float(recall),
                "time": float(time_cost),
                "FPS": float(fps),
            }
        )

        table_data = [header] + list(zip(*table_columns))
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log("\n" + table.table, logger=logger)

        return ret_dict


@METRICS.register_module()
class SscMetric(BaseSscMetric):
    """Compute metrics for semantic segmentation completion (SS)."""

    def __init__(
        self,
        collect_device: str = "cpu",
        prefix: Optional[str] = None,
        num_classes: int = 20,
        free_index: int = 0,
        ignore_index: int = 255,
        **kwargs,
    ):
        super().__init__(
            collect_device=collect_device,
            prefix=prefix,
            num_classes=num_classes,
            free_index=free_index,
            ignore_index=ignore_index,
            **kwargs,
        )

    def compute_metrics(self, results: list) -> Dict[str, float]:

        label2cat = self.dataset_meta["label2cat"]
        ignore_index = self.dataset_meta.get("ignore_index", self.ignore_index)
        stats = self.ssc_compute.get_stats()
        result_dict = self.log_show(
            stats, label2cat, ignore_index, logger=MMLogger.get_current_instance()
        )
        self.ssc_compute.reset()

        return result_dict


@METRICS.register_module()
class FPSMetric(BaseMetric):
    """Compute FPS metric for semantic segmentation."""

    def __init__(
        self, collect_device: str = "cpu", prefix: Optional[str] = None, **kwargs
    ):
        super().__init__(prefix=prefix, collect_device=collect_device)
        self.time_cost = []

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Store the time cost for FPS computation."""
        for data_sample in data_samples:
            self.time_cost.append(data_sample["time_cost"])

    def log_show(self, logger=None) -> Dict[str, float]:
        """Display FPS in a table."""
        time = np.mean(self.time_cost)
        fps = 1 / time

        ret_dict = {"time": float(time), "FPS": float(fps)}

        table_data = [
            ["Metric", "time", "FPS"],
            ["results", f"{time:.4f}", f"{fps:.4f}"],
        ]
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log("\n" + table.table, logger=logger)

        return ret_dict

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute and return FPS metric."""
        results_dict = self.log_show(logger=MMLogger.get_current_instance())
        self.time_cost = []
        return results_dict
