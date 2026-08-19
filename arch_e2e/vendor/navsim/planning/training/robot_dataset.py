"""Load preprocessed robot .npz samples (see robot_data_prep) for DiffusionDrive training."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig


class RobotNpzDataset(Dataset):
    """
    Reads manifest.csv produced by robot_data_prep (columns: idx, npz, ...).
    Each npz has: image_u8, lidar_feature, trajectory, status, ...
    """

    def __init__(
        self,
        manifest_csv: Path,
        config: TransfuserConfig,
        *,
        use_auxiliary_supervision: bool = False,
    ) -> None:
        super().__init__()
        self._config = config
        self._root = manifest_csv.resolve().parent
        self._use_aux = use_auxiliary_supervision
        self._to_tensor = transforms.ToTensor()

        self._rows: List[Dict[str, str]] = []
        with open(manifest_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._rows.append(row)
        if not self._rows:
            raise RuntimeError(f"No rows in {manifest_csv}")

        self._num_poses = config.trajectory_sampling.num_poses
        h, w = config.bev_semantic_frame
        self._bev_dummy = torch.zeros((h, w), dtype=torch.long)
        self._agent_states_dummy = torch.zeros(
            (config.num_bounding_boxes, 5),
            dtype=torch.float32,
        )
        self._agent_labels_dummy = torch.zeros(
            (config.num_bounding_boxes,),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        rel = self._rows[idx]["npz"]
        path = self._root / rel
        z = np.load(path, allow_pickle=False)

        image_u8 = z["image_u8"]
        lidar = np.asarray(z["lidar_feature"], dtype=np.float32)
        traj = np.asarray(z["trajectory"], dtype=np.float32)
        status = np.asarray(z["status"], dtype=np.float32)

        if traj.shape[0] != self._num_poses or traj.shape[1] != 3:
            raise ValueError(
                f"{path}: trajectory shape {traj.shape}, expected ({self._num_poses}, 3). "
                "Match preprocessing time_horizon_s / interval_length_s to TransfuserConfig.trajectory_sampling."
            )

        features: Dict[str, torch.Tensor] = {
            "camera_feature": self._to_tensor(image_u8),
            "lidar_feature": torch.from_numpy(lidar),
            "status_feature": torch.from_numpy(status),
        }

        targets: Dict[str, torch.Tensor] = {"trajectory": torch.from_numpy(traj)}
        if status.shape[0] >= 4:
            mode_idx = int(np.argmax(status[:4]))
            targets["mode_label"] = torch.tensor(mode_idx, dtype=torch.long)
        # Robot recordings have no map/agents; only attach dummy BEV/agent targets if you enable auxiliary loss.
        if self._use_aux:
            targets["agent_states"] = self._agent_states_dummy.clone()
            targets["agent_labels"] = self._agent_labels_dummy.clone()
            targets["bev_semantic_map"] = self._bev_dummy.clone()

        return features, targets
