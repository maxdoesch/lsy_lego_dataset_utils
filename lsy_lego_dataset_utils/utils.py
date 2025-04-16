import numpy as np
import cv2
import os
import json
import shutil
import h5py
import yaml
from datetime import datetime
from typing import Dict, Any, List

def duplicate_dataset(old_dataset_path: str, new_dataset_path: str):
    """Creates a duplicate of the dataset in before making changes."""
    if os.path.exists(new_dataset_path):
        shutil.rmtree(new_dataset_path)
    shutil.copytree(old_dataset_path, new_dataset_path)
    print(f"Duplicate created at {new_dataset_path}")

def load_metadata(episode_path: str) -> dict:
    metadata_path = os.path.join(episode_path, 'metadata.json')
    
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata

def dump_metadata(episode_path: str, metadata: dict):
    metadata_path = os.path.join(episode_path, 'metadata.json')

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

def save_metadata(self, path, steps_per_episode):
        metadata = {
            "episode_steps": steps_per_episode,
            "logging_frequency": self.logging_frequency,
            "data_collector": self.data_collector,
            "collection_time": datetime.now(ZoneInfo("Europe/Berlin")).strftime("%Y-%m-%d %H:%M:%S")
        }
    
        with open(path, "w") as f:
            json.dump(metadata, f, indent=4)

def resize_with_aspect_ratio(image: np.ndarray, target_res: tuple):
    """
    Resize an image to fit within a target resolution while maintaining aspect ratio,
    cropping if necessary.
    
    :param image: Input image as a numpy array (HxWxC)
    :param target_res: Tuple (target_height, target_width)
    :return: Resized and cropped image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_res
    
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    
    cropped_image = resized[start_y:start_y + target_h, start_x:start_x + target_w]
    
    return cropped_image

def pad_to_match(img1, img2):
    """Pads img1 and img2 so that their height and width match."""
    def get_hw(img):
        return img.shape[1:3] if img.ndim == 4 else img.shape[:2]

    def pad(img, target_h, target_w):
        h, w = get_hw(img)
        pad_h, pad_w = target_h - h, target_w - w
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2

        if img.ndim == 4:
            padding = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        elif img.ndim == 3:
            padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else:
            padding = ((pad_top, pad_bottom), (pad_left, pad_right))
        
        return np.pad(img, padding, mode='constant', constant_values=255)

    h1, w1 = get_hw(img1)
    h2, w2 = get_hw(img2)
    target_h, target_w = max(h1, h2), max(w1, w2)

    return pad(img1, target_h, target_w), pad(img2, target_h, target_w)

def save_h5_file(h5_file_path: str, dataset_dict: dict):
    with h5py.File(h5_file_path, 'w') as f:
            for step_idx, step in enumerate(dataset_dict['steps']):
                group = f.create_group(f"step_{step_idx}")

                for category, category_data in step.items():
                    if isinstance(category_data, dict):
                        subgroup = group.create_group(category)
                        for key, value in category_data.items():
                            subgroup.create_dataset(key, data=value)
                    else:
                        group.create_dataset(category, data=category_data)


def print_h5_file(file_path):
    """
    Prints the contents of an HDF5 file to the console.
    Recursively prints groups and datasets in the file.
    
    :param file_path: Path to the .h5 file to be printed
    """
    def print_group(group, indent=0):
        """
        Recursively print a group and its contents.
        
        :param group: The current group to print
        :param indent: The current indentation level
        """
        space = ' ' * indent
        print(f"{space}Group: {group.name}")
        
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                print_group(item, indent + 2)
            elif isinstance(item, h5py.Dataset):
                print(f"{space}  Dataset: {key} (shape: {item.shape}, dtype: {item.dtype})")
                
                if item.shape == ():
                    print(f"{space}    Data: {item[()]}")
                else:
                    print(f"{space}    Data: {item[:5]}...")

    try:
        with h5py.File(file_path, 'r') as f:
            print_group(f)
    except Exception as e:
        print(f"Error reading the HDF5 file: {e}")

def save_mp4_video(video_folder: str, video_name: str, images: List[np.ndarray], fps=10):
    """
    Saves the given images as an MP4 video.
    
    :param video_folder: Folder where the video will be saved
    :param video_name: Name of the video file (e.g., 'wrist_video.mp4')
    :param images: List of images (rgb8 format) to be saved in the video
    :param fps: Frames per second for the video
    """
    video_path = os.path.join(video_folder, video_name)
    
    frame_height, frame_width, _ = images[0].shape
    frame_size = (frame_width, frame_height)

    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
    
    for image in images:
        video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    video_writer.release()


def resize_with_aspect_ratio(image: np.ndarray, target_res: tuple):
    """
    Resize an image to fit within a target resolution while maintaining aspect ratio,
    cropping if necessary.
    
    :param image: Input image as a numpy array (HxWxC)
    :param target_res: Tuple (target_height, target_width)
    :return: Resized and cropped image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_res
    
    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    
    cropped_image = resized[start_y:start_y + target_h, start_x:start_x + target_w]
    
    return cropped_image

def save_config(dataset_path: str, config: Dict[str, Any]) -> None:
    """Saves a configuration dictionary as a YAML file in the given dataset path."""
    try:
        os.makedirs(dataset_path, exist_ok=True)
        config_path = os.path.join(dataset_path, "config.yaml")
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
    except (OSError, yaml.YAMLError) as e:
        print(f"Error saving config: {e}")

def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a configuration dictionary from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        print(f"Error loading config: {e}")
        return {}