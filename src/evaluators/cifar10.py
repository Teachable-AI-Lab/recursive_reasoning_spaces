from typing import Dict, Optional

import torch
import torch.distributed as dist

from dataset.common import PuzzleDatasetMetadata
from models.losses import IGNORE_LABEL_ID


class CIFAR10:
    required_outputs = {"logits", "labels"}

    def __init__(self, data_path: str, eval_metadata: PuzzleDatasetMetadata):
        super().__init__()
        self.data_path = data_path
        self.eval_metadata = eval_metadata
        self._local_correct = 0
        self._local_total = 0
        self._local_class_only_correct = 0
        self._local_pred_in_class_range = 0

    def begin_eval(self):
        self._local_correct = 0
        self._local_total = 0
        self._local_class_only_correct = 0
        self._local_pred_in_class_range = 0

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        logits = preds["logits"]
        labels = batch["labels"]

        class_targets = labels[:, 0]
        valid_mask = class_targets != IGNORE_LABEL_ID

        if valid_mask.any():
            token_preds = logits[:, 0, :].argmax(dim=-1)
            class_only_preds = logits[:, 0, 257:267].argmax(dim=-1) + 257

            self._local_correct += int((token_preds[valid_mask] == class_targets[valid_mask]).sum().item())
            self._local_class_only_correct += int((class_only_preds[valid_mask] == class_targets[valid_mask]).sum().item())
            self._local_pred_in_class_range += int(((token_preds[valid_mask] >= 257) & (token_preds[valid_mask] < 267)).sum().item())
            self._local_total += int(valid_mask.sum().item())

    def result(
        self,
        save_path: Optional[str],
        rank: int,
        world_size: int,
        group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> Optional[Dict[str, float]]:
        if world_size == 1 or not dist.is_available() or not dist.is_initialized():
            if rank != 0:
                return None
            if self._local_total == 0:
                return {
                    "CIFAR10/top1": 0.0,
                    "CIFAR10/top1_class_only": 0.0,
                    "CIFAR10/pred_in_class_range": 0.0,
                }
            return {
                "CIFAR10/top1": self._local_correct / self._local_total,
                "CIFAR10/top1_class_only": self._local_class_only_correct / self._local_total,
                "CIFAR10/pred_in_class_range": self._local_pred_in_class_range / self._local_total,
            }

        global_stats = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(
            (
                self._local_correct,
                self._local_total,
                self._local_class_only_correct,
                self._local_pred_in_class_range,
            ),
            global_stats,
            dst=0,
            group=group,
        )

        if rank != 0:
            return None

        total_correct = sum(x[0] for x in global_stats)  # type: ignore
        total_examples = sum(x[1] for x in global_stats)  # type: ignore
        total_class_only_correct = sum(x[2] for x in global_stats)  # type: ignore
        total_pred_in_class_range = sum(x[3] for x in global_stats)  # type: ignore

        if total_examples == 0:
            return {
                "CIFAR10/top1": 0.0,
                "CIFAR10/top1_class_only": 0.0,
                "CIFAR10/pred_in_class_range": 0.0,
            }

        return {
            "CIFAR10/top1": total_correct / total_examples,
            "CIFAR10/top1_class_only": total_class_only_correct / total_examples,
            "CIFAR10/pred_in_class_range": total_pred_in_class_range / total_examples,
        }
