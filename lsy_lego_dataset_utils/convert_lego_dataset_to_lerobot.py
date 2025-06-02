"""
Script to convert Lego Dataset hdf5 data to the LeRobot dataset format.

Adapted from https://github.com/Physical-Intelligence/openpi/blob/main/examples/aloha_real/convert_aloha_data_to_lerobot.py

Example usage:
python data_collector/scripts/convert_lego_dataset_to_lerobot.py \
    --dataset_path=/path/to/raw/data \
    --repo_id=<org>/<dataset-name> \
    --task=lego_task \
    --robot_type=franka
"""

import os
from pathlib import Path
import shutil

from absl import app, flags
import json
import h5py
import numpy as np
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

FLAGS = flags.FLAGS

flags.DEFINE_string("dataset_path", None, "Path to the raw dataset directory.")
flags.DEFINE_string("repo_id", "lego_dataset", "Hugging Face repo ID to save the dataset under.")
flags.DEFINE_string("task", "lego_task", "Task name to associate with the dataset.")
flags.DEFINE_string("robot_type", "franka", "Type of robot (e.g., franka).")

flags.mark_flag_as_required("dataset_path")


class LegoDatasetConverter:
    def __init__(self, dataset_path: str, repo_id: str, robot_type: str):

        self.dataset_path = dataset_path
        self.repo_id = repo_id
        self.robot_type = robot_type

        self.episode_paths = [
            os.path.join(self.dataset_path, f)
            for f in os.listdir(self.dataset_path)
            if f.startswith('episode_')
        ]
        if not self.episode_paths:
            raise FileNotFoundError("No episodes found in dataset directory.")

        self.cameras = {
            "third_person_image": (256, 256, 3),
            "wrist_image": (256, 256, 3),
        }

        self.dimensions = [
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
            "gripper",
        ]

    def create_empty_dataset(self) -> LeRobotDataset:
        metadata_path = os.path.join(self.episode_paths[0], "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.json not found in {self.episode_paths[0]}")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            fps = metadata.get("logging_frequency", 10)  # fallback to 10 if not found

        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.dimensions),),
                "names": [self.dimensions],
            },
            "action": {
                "dtype": "float32",
                "shape": (len(self.dimensions),),
                "names": [self.dimensions],
            },
        }

        for cam, res in self.cameras.items():
            features[f"observation.images.{cam}"] = {
                "dtype": 'image',
                "shape": res,
                "names": ["height", "width", "channels"],
            }

        if Path(HF_LEROBOT_HOME / self.repo_id).exists():
            shutil.rmtree(HF_LEROBOT_HOME / self.repo_id)

        return LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=fps,
            robot_type=self.robot_type,
            features=features,
            use_videos=False,
        )

    def populate_dataset(self, dataset: LeRobotDataset, task: str) -> LeRobotDataset:
        for ep_path in tqdm.tqdm(self.episode_paths):
            trajectory_path = os.path.join(ep_path, 'trajectory.h5')

            with h5py.File(trajectory_path, "r") as episode:
                steps = [f'step_{i}' for i in range(len(episode)) if f'step_{i}' in episode]
                if not steps:
                    raise ValueError("No steps found in episode data.")

                for step in steps:
                    gripper_position = episode[step]['observation']['gripper_position'][()]
                    gripper_position_array = np.array([gripper_position], dtype=np.float32)
                    
                    if 'gripper' in episode[step]['action']:
                        gripper_position_binarized = episode[step]['action']['gripper'][()]
                        gripper_position_binarized_array = np.array([gripper_position_binarized], dtype=np.float32)
                    else:
                        gripper_position_binarized = 1 if gripper_position > 0.1 else 0
                        gripper_position_binarized_array = np.array([gripper_position_binarized], dtype=np.float32)

                    frame = {
                        "observation.state": np.concatenate(
                            (
                                episode[step]["observation"]["cartesian_position"][()],
                                gripper_position_array,
                            ),
                            dtype=np.float32,
                        ),
                        "action": np.concatenate(
                            (
                                episode[step]["action"]["cartesian_position_delta"][()],
                                gripper_position_binarized_array,
                            ),
                            dtype=np.float32,
                        ),
                        "task": task,
                    }

                    for cam in self.cameras.keys():
                        frame[f"observation.images.{cam}"] = episode[step]["observation"][cam][()]

                    dataset.add_frame(frame)

            dataset.save_episode()

        return dataset


def main(_):
    converter = LegoDatasetConverter(
        dataset_path=FLAGS.dataset_path,
        repo_id=FLAGS.repo_id,
        robot_type=FLAGS.robot_type,
    )

    dataset = converter.create_empty_dataset()
    converter.populate_dataset(dataset, task=FLAGS.task)


if __name__ == "__main__":
    app.run(main)
