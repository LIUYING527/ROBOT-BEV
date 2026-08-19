"""Train DiffusionDrive (Transfuser) on robot_data_prep NPZ manifests (camera + LiDAR BEV + trajectory)."""

from __future__ import annotations

import math
import logging
from pathlib import Path
import random
from typing import Tuple

import hydra
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Subset

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.robot_dataset import RobotNpzDataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "robot_training"


def _split_dataset_indices(n: int, val_ratio: float, seed: int) -> Tuple[list[int], list[int]]:
    if n < 2:
        raise RuntimeError("Need at least 2 samples to split train/val.")
    ratio = max(0.0, min(0.5, float(val_ratio)))
    val_n = int(math.ceil(n * ratio))
    val_n = max(1, min(n - 1, val_n))
    indices = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(indices)
    val_idx = sorted(indices[:val_n])
    train_idx = sorted(indices[val_n:])
    return train_idx, val_idx


def build_robot_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    train_m = Path(cfg.train_manifest)
    if not train_m.is_file():
        raise FileNotFoundError(train_m)

    aux = bool(getattr(agent._config, "use_auxiliary_supervision", True))
    if cfg.val_manifest in (None, "", "null"):
        full_ds = RobotNpzDataset(train_m, agent._config, use_auxiliary_supervision=aux)
        train_idx, val_idx = _split_dataset_indices(
            len(full_ds),
            float(getattr(cfg, "val_ratio", 0.1)),
            int(getattr(cfg, "split_seed", 0)),
        )
        train_ds = Subset(full_ds, train_idx)
        val_ds = Subset(full_ds, val_idx)
        logger.info(
            "Auto split from one manifest: train=%d, val=%d (val_ratio=%.3f, split_seed=%d)",
            len(train_ds),
            len(val_ds),
            float(getattr(cfg, "val_ratio", 0.1)),
            int(getattr(cfg, "split_seed", 0)),
        )
    else:
        val_m = Path(cfg.val_manifest)
        if not val_m.is_file():
            raise FileNotFoundError(val_m)
        train_ds = RobotNpzDataset(train_m, agent._config, use_auxiliary_supervision=aux)
        val_ds = RobotNpzDataset(val_m, agent._config, use_auxiliary_supervision=aux)
    return train_ds, val_ds


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)
    logger.info("Global seed set to %s", cfg.seed)
    logger.info("Results directory: %s", cfg.output_dir)

    logger.info("Building agent")
    try:
        agent: AbstractAgent = instantiate(cfg.agent)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Failed to build DiffusionDrive agent. "
            "Please ensure plan anchors exist and set PLAN_ANCHOR_PATH to kmeans_navsim_traj_20.npy "
            "(shape should be [20, 8, 2])."
        ) from e

    logger.info("Building Lightning module")
    lightning_module = AgentLightningModule(agent=agent)

    logger.info("Building robot NPZ datasets")
    train_data, val_data = build_robot_datasets(cfg, agent)

    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num training samples: %d", len(train_data))
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building trainer")
    callbacks = list(agent.get_training_callbacks())
    callbacks.append(
        ModelCheckpoint(
            monitor="val/loss_epoch",
            mode="min",
            save_top_k=1,
            filename="best-val-loss",
            auto_insert_metric_name=False,
        )
    )
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks)

    logger.info("Starting training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
