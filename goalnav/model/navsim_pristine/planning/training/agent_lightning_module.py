import pytorch_lightning as pl
import torch
import logging

from torch import Tensor
from typing import Dict, Tuple

from navsim.agents.abstract_agent import AbstractAgent

logger = logging.getLogger(__name__)


class AgentLightningModule(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self, agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()
        self.agent = agent

    @staticmethod
    def _trajectory_metrics(prediction: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if "trajectory" not in prediction or "trajectory" not in targets:
            return {}
        pred = prediction["trajectory"]
        gt = targets["trajectory"]
        if pred.ndim != 3 or gt.ndim != 3 or pred.shape[-1] < 2 or gt.shape[-1] < 2:
            return {}

        pos_err = torch.linalg.norm(pred[..., :2] - gt[..., :2], dim=-1)  # [B, T]
        ade = pos_err.mean()
        fde = pos_err[:, -1].mean()
        return {"trajectory_ADE_m": ade, "trajectory_FDE_m": fde}

    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets = batch
        prediction = self.agent.forward(features, targets)
        # loss = self.agent.compute_loss(features, targets, prediction)
        # self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        # return loss
        loss_dict = self.agent.compute_loss(features, targets, prediction)
        batch_size = int(features["camera_feature"].shape[0]) if "camera_feature" in features else None
        for k, v in loss_dict.items():
            if v is not None:
                self.log(
                    f"{logging_prefix}/{k}",
                    v,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )

        metric_dict = self._trajectory_metrics(prediction, targets)
        for k, v in metric_dict.items():
            self.log(
                f"{logging_prefix}/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=(logging_prefix == "val"),
                sync_dist=True,
                batch_size=batch_size,
            )
        return loss_dict['loss']

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """Print key validation metrics explicitly in terminal logs."""
        metrics = self.trainer.callback_metrics
        show_keys = (
            "val/loss_epoch",
            "val/trajectory_loss_epoch",
            "val/mode_loss_epoch",
            "val/mode_acc_epoch",
            "val/trajectory_ADE_m",
            "val/trajectory_FDE_m",
        )
        pretty = []
        for k in show_keys:
            v = metrics.get(k, None)
            if v is not None:
                pretty.append(f"{k}={float(v):.6f}")
        if pretty:
            logger.info("Validation epoch end: %s", ", ".join(pretty))

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()
