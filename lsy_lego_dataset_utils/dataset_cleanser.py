import os
import shutil
import h5py
import numpy as np
import argparse
import json
import ast
import utils

class DatasetCleanser():
    def __init__(self, configs: dict):

        for dataset_path in ['dataset_path_original', 'dataset_path_modified']:
            if (dataset_path not in configs):
                raise ValueError(f"No '{dataset_path}' found in configs.")
            
        self.dataset_path = configs['dataset_path_original']
        self.dataset_path_modified = configs['dataset_path_modified']

        self._create_job_list(configs)

    def _create_job_list(self, configs: dict):
        self.job_list = [(self._check_folder_structure, {})]

        if 'shift_gripper' in configs:
            self.job_list.append((self._shift_gripper, configs['shift_gripper']))
        if 'remove_inactivity' in configs:
            self.job_list.append((self._remove_inactivity, {}))
            self.job_list.append((self._rename_steps, {}))
        if 'subsample_episode' in configs:
            self.job_list.append((self._subsample_episode, configs['subsample_episode']))
            self.job_list.append((self._rename_steps, {}))
        if 'relabel_episode' in configs:
            self.job_list.append((self._relabel_episode, configs['relabel_episode']))
        if 'resize_observation' in configs:
            self.job_list.append((self._resize_observation, configs['resize_observation']))
        if 'split_dataset' in configs:
            self.job_list.append((self._split_dataset_by_instruction, {}))

    def _shift_gripper(self, episode_path: str, args: dict):
        shift_steps = args['shift_steps']
        trajectory_path = os.path.join(episode_path, 'trajectory.h5')
        
        with h5py.File(trajectory_path, 'r+') as episode:
            steps = sorted([key for key in episode.keys() if key.startswith('step_')], 
                        key=lambda x: int(x.split('_')[1]))
            if not steps:
                raise ValueError("No steps found in episode data.")
            
            gripper_positions = np.array([episode[step]['observation']['gripper_position'][()] for step in steps])
            gripper_action = np.array([episode[step]['action']['gripper'][()] for step in steps])

            if shift_steps > 0:
                gripper_positions = np.concatenate(
                    (np.tile(gripper_positions[0:1], (shift_steps)), gripper_positions[:-shift_steps])
                )
                gripper_action = np.concatenate(
                    (np.tile(gripper_action[0:1], (shift_steps)), gripper_action[:-shift_steps])
                )
            elif shift_steps < 0:
                gripper_positions = np.concatenate(
                    (gripper_positions[-shift_steps:], np.tile(gripper_positions[-1:], (-shift_steps)))
                )
                gripper_action = np.concatenate(
                    (gripper_action[-shift_steps:], np.tile(gripper_action[-1:], (-shift_steps)))
                )

            for step in steps:
                episode[step]['observation']['gripper_position'][()] = gripper_positions[steps.index(step)]
                episode[step]['action']['gripper'][()] = gripper_action[steps.index(step)]

    def _resize_observation(self, episode_path: str, args: dict):
        target_resolutions = args['target_resolution']
        trajectory_path = os.path.join(episode_path, 'trajectory.h5')

        with h5py.File(trajectory_path, 'r+') as episode:
            steps = [f'step_{i}' for i in range(len(episode)) if f'step_{i}' in episode]
            if not steps:
                raise ValueError("No steps found in episode data.")

            for step in steps:
                for cam_name, target_size in target_resolutions.items():
                    if cam_name in episode[step]['observation']:
                        image = episode[step]['observation'][cam_name][()]
                        resized_image = utils.resize_with_aspect_ratio(image, target_size)
                        del episode[step]['observation'][cam_name]
                        episode[step]['observation'].create_dataset(cam_name, data=resized_image)
                    else:
                        print(f"Camera {cam_name} not found in step {step}. Skipping resizing.")
            
    def _remove_inactivity_end_beginning(self, episode_path: str, args: dict):
        trajectory_path = os.path.join(episode_path, 'trajectory.h5')

        with h5py.File(trajectory_path, 'r+') as episode:  # Open in 'r+' mode to read and modify the file
            steps = [f'step_{i}' for i in range(len(episode)) if f'step_{i}' in episode]
            if not steps:
                raise ValueError("No steps found in episode data.")

            actions_joint = np.array([episode[step]['action']['joint_position_delta'][()] for step in steps])

            threshold = 1e-3

            start_index = 0
            while start_index < len(actions_joint) and np.all(np.abs(actions_joint[start_index]) < threshold):
                start_index += 1
            
            end_index = len(actions_joint) - 1
            while end_index >= 0 and np.all(np.abs(actions_joint[end_index]) < threshold):
                end_index -= 1

            active_steps = steps[start_index:end_index+1]

            for step in list(episode.keys()):
                if step not in active_steps:
                    del episode[step]

    def _remove_inactivity(self, episode_path: str, args: dict):
        trajectory_path = os.path.join(episode_path, 'trajectory.h5')
        
        with h5py.File(trajectory_path, 'r+') as episode:
            steps = [f'step_{i}' for i in range(len(episode)) if f'step_{i}' in episode]
            if not steps:
                raise ValueError("No steps found in episode data.")

            actions_joint = np.array([episode[step]['action']['joint_position_delta'][()] for step in steps])
            gripper_positions = np.array([episode[step]['observation']['gripper_position'][()] for step in steps])
            gripper_actions = np.array([episode[step]['action']['gripper'][()] for step in steps])
            
            threshold = 1e-3
            active_steps = []

            for i in range(len(steps)):
                is_action_active = not np.all(np.abs(actions_joint[i]) < threshold)
                is_gripper_moving = i == (len(gripper_positions) - 1) or not np.allclose(gripper_positions[i+1], gripper_positions[i], atol=threshold)
                is_gripper_action_active = i == (len(gripper_actions) - 1) or not np.allclose(gripper_actions[i+1], gripper_actions[i], atol=threshold)
                
                if is_action_active or is_gripper_moving or is_gripper_action_active:
                    active_steps.append(steps[i])
            
            for step in list(episode.keys()):
                if step not in active_steps:
                    del episode[step]
            
            print(f"Removed {len(steps) - len(active_steps)} inactive steps from {episode_path}.")

    def _subsample_episode(self, episode_path: str, args: dict):
        """Reduce the number of steps in a given episode."""
        target_frequency = args['target_frequency']
        metadata = utils.load_metadata(episode_path)
        
        original_frequency = metadata.get('logging_frequency', None)
        if original_frequency is None:
            raise ValueError("Original logging frequency not found in metadata.")
        
        if target_frequency is None or target_frequency >= original_frequency or original_frequency % target_frequency != 0:
            raise ValueError("Target frequency must be a smaller integer fraction of the original frequency.")
        
        subsample_factor = original_frequency // target_frequency
        
        trajectory_path = os.path.join(episode_path, 'trajectory.h5')
        
        with h5py.File(trajectory_path, 'r+') as episode:
            steps = sorted([key for key in episode.keys() if key.startswith('step_')], 
                        key=lambda x: int(x.split('_')[1]))
            
            if not steps:
                raise ValueError("No steps found in episode data.")
            
            selected_steps = []
            
            for idx in range(0, len(steps), subsample_factor):
                start_idx = idx
                end_idx = min(idx + subsample_factor, len(steps))
                
                summed_cartesian = np.sum(
                    [episode[steps[i]]['action']['cartesian_position_delta'][()] for i in range(start_idx, end_idx)], axis=0
                )
                summed_joint = np.sum(
                    [episode[steps[i]]['action']['joint_position_delta'][()] for i in range(start_idx, end_idx)], axis=0
                )
                
                del episode[steps[start_idx]]['action']['cartesian_position_delta']
                episode[steps[start_idx]]['action'].create_dataset('cartesian_position_delta', data=summed_cartesian)

                del episode[steps[start_idx]]['action']['joint_position_delta']
                episode[steps[start_idx]]['action'].create_dataset('joint_position_delta', data=summed_joint)
                
                selected_steps.append(steps[start_idx])
            
            for step in list(episode.keys()):
                if step not in selected_steps:
                    del episode[step]

        metadata['logging_frequency'] = target_frequency
        utils.dump_metadata(episode_path, metadata)

    def _relabel_episode(self, episode_path: str, args: dict):
        """Relabel the episode instruction with given label."""
        new_instruction_labels = args['new_instruction_labels']

        trajectory_path = os.path.join(episode_path, 'trajectory.h5')
        
        with h5py.File(trajectory_path, 'r+') as episode:
            steps = sorted([key for key in episode.keys() if key.startswith('step_')], 
                        key=lambda x: int(x.split('_')[1]))
            
            if not steps:
                raise ValueError("No steps found in episode data.")
            
            instruction_idx = np.random.randint(0, len(new_instruction_labels))
            for step in steps:
                episode[step]['instruction'][()] = new_instruction_labels[instruction_idx]

    def _rename_steps(self, episode_path, args: dict):
        trajectory_path = os.path.join(episode_path, 'trajectory.h5')
        temp_path = os.path.join(episode_path, 'trajectory_temp.h5')
        
        with h5py.File(trajectory_path, 'r') as episode, h5py.File(temp_path, 'w') as temp_episode:
            steps = sorted([key for key in episode.keys() if key.startswith('step_')], 
                        key=lambda x: int(x.split('_')[1]))
            
            if not steps:
                raise ValueError("No steps found in episode data.")
            
            for new_idx, old_name in enumerate(steps):
                new_name = f'step_{new_idx}'
                episode.copy(old_name, temp_episode, new_name)
            
            for key in episode.keys():
                if key not in steps:
                    episode.copy(key, temp_episode, key)

            episode_len = len([key for key in temp_episode.keys() if key.startswith('step_')])
        
        os.replace(temp_path, trajectory_path)

        metadata = utils.load_metadata(episode_path)
        metadata['episode_steps'] = episode_len
        utils.dump_metadata(episode_path, metadata)

    def _check_folder_structure(self, episode_path: str, args: dict):
        """Checks whether the episode folder exists and contains the required `trajectory.h5` and `metadata.json` file."""
        if not os.path.isdir(episode_path):
            raise FileNotFoundError(f"Episode folder not found: {episode_path}")
        
        metadata_path = os.path.join(episode_path, 'metadata.json')
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Missing file: {metadata_path}")
        
        trajectory_file = os.path.join(episode_path, 'trajectory.h5')
        if not os.path.isfile(trajectory_file):
            raise FileNotFoundError(f"Missing file: {trajectory_file}")
        
    def _split_dataset_by_instruction(self, episode_path: str, args: dict):

        try:
            trajectory_path = os.path.join(episode_path, 'trajectory.h5')
            with h5py.File(trajectory_path, 'r') as episode:
                steps = sorted(
                    [key for key in episode.keys() if key.startswith('step_')],
                    key=lambda x: int(x.split('_')[1])
                )
                if not steps:
                    raise ValueError("No steps found.")

                instruction = episode[steps[0]]['instruction'][()].decode() \
                    if isinstance(episode[steps[0]]['instruction'][()], bytes) \
                    else episode[steps[0]]['instruction'][()]

        except Exception as e:
            print(f"Could not read instruction from {episode_path}: {e}")

        dest_dir = os.path.join(self.dataset_path_modified, f'dataset_{instruction}')
        os.makedirs(dest_dir, exist_ok=True)

        new_ep_idx = len([f for f in os.listdir(dest_dir) if f.startswith('episode_')])
        dest_path = os.path.join(dest_dir, f"episode_{new_ep_idx}")
        shutil.move(episode_path, dest_path)

        print(f"Added episode to subfolder: {self.dataset_path_modified}.")
        
    def _reorder_episodes(self, dataset_path: str):
        """Reorders episode folders to fill in missing indices."""
        episode_indices = sorted(int(f.split('_')[1]) for f in os.listdir(dataset_path) if f.startswith('episode_'))
        for i, episode_index in enumerate(episode_indices):
            expected_name = f"episode_{i}"
            current_name = f"episode_{episode_index}"
            if expected_name != current_name:
                os.rename(os.path.join(dataset_path, current_name), os.path.join(dataset_path, expected_name))

    def combine_datasets(self, dataset_path: str):
        """
        Flattens nested datasets by moving all episode folders from subdirectories
        under dataset_path_original into dataset_path_modified as a single sequence.
        """             
        subfolders = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.startswith("dataset_") and os.path.isdir(os.path.join(dataset_path, f))]

        if len(subfolders) == 0:
            return

        episode_counter = 0
        for subfolder in subfolders:
            episodes = [os.path.join(subfolder, f) for f in os.listdir(subfolder) if f.startswith("episode_") and os.path.isdir(os.path.join(subfolder, f))]

            for ep in sorted(episodes):
                new_name = f"episode_{episode_counter}"
                new_path = os.path.join(dataset_path, new_name)
                shutil.copytree(ep, new_path)
                episode_counter += 1
            
            shutil.rmtree(subfolder)

        print(f"Combined datasets into {self.dataset_path_modified} with {episode_counter} episodes.")

    def check_episode(self, episode_path: str):
        """Checks the folder structure and deletes the episode if it is invalid."""
        try:
            for job_fcn, job_arg in self.job_list:
                job_fcn(episode_path, job_arg)
            print(f"Episode valid: {episode_path}")
        except Exception as e:
            print(f"Error processing episode {episode_path}: {e} - Deleting...")
            shutil.rmtree(episode_path)

    def check_all_episodes(self):
        """Checks all episodes in the dataset and removes invalid ones."""
        utils.duplicate_dataset(self.dataset_path, self.dataset_path_modified)

        self.combine_datasets(self.dataset_path_modified)

        episode_paths = [os.path.join(self.dataset_path_modified, f) for f in os.listdir(self.dataset_path_modified) if f.startswith('episode_')]
        if not episode_paths:
            raise FileNotFoundError("No episodes found in dataset directory.")

        for episode_path in episode_paths:
            self.check_episode(episode_path)

        self._reorder_episodes(self.dataset_path_modified)

class CreateJob():
    def __init__(self):
        workspace_path = os.getcwd()
        self.dataset_path_original = os.path.join(workspace_path, 'dataset')
    
    def create_job_from_args(self, args) -> DatasetCleanser:
        if args.dataset_path is not None:
            self.dataset_path_original = args.dataset_path

        self.dataset_path_original = self.dataset_path_original.rstrip('/')
        
        base_dir = os.path.dirname(self.dataset_path_original)
        base_name = os.path.basename(self.dataset_path_original)
        modified_name = f"{base_name}_modified"
        dataset_path_modified = os.path.join(base_dir, modified_name)

        config = {
            'dataset_path_original': self.dataset_path_original,
            'dataset_path_modified': dataset_path_modified,
        }
        if args.split_dataset is not None and args.split_dataset:
            config['split_dataset'] = {}
        if args.shift_gripper is not None:
            config['shift_gripper'] = {'shift_steps': args.shift_gripper}
        if args.inactivity is not None and args.inactivity:
            config['remove_inactivity'] = {}
        if args.subsample is not None:
            config['subsample_episode'] = {'target_frequency': args.subsample}
        if args.relabel is not None:
            config['relabel_episode'] = {'new_instruction_labels': args.relabel}
        if args.resize is not None:
            try:
                resize_observation = ast.literal_eval(args.resize)
                if not isinstance(resize_observation, dict):
                    raise ValueError("resize_observation must be a dictionary.")
                for cam, dims in resize_observation.items():
                    if not (isinstance(dims, tuple) and len(dims) == 2):
                        raise ValueError(f"Invalid size for camera '{cam}': must be (width, height)")
                config['resize_observation'] = {'target_resolution': resize_observation}
            except Exception as e:
                raise ValueError(f"Failed to parse --resize: {e}")

        return DatasetCleanser(config)
    
def main():
    parser = argparse.ArgumentParser(description="Posprocessing of lego dataset")
    parser.add_argument("--inactivity", action='store_true', required=False, help="Remove steps where actions are inactive.")
    parser.add_argument("--subsample", type=int, required=False, help='Reduce logging frequency to given value. (Must be smaller and a fraction of origingal log frequency)')
    parser.add_argument("--relabel", type=str, nargs='+', required=False, help="Relabel instructions with the given label(s).")
    parser.add_argument("--dataset_path", type=str, default=None, required=False, help="Path to the dataset directory.")
    parser.add_argument("--resize", type=str, required=False, help="Resize camera images. Syntax: --resize \"{'cam_name': (width, height), ...}\"")
    parser.add_argument("--shift_gripper", type=int, default=0, required=False, help="Shift gripper position by given time steps.")
    parser.add_argument('--split_dataset', action='store_true', required=False, help="Split the dataset based on the language instruction")

    args = parser.parse_args()
    dataset_cleanser = CreateJob().create_job_from_args(args)
    dataset_cleanser.check_all_episodes()

if __name__ == "__main__":
    main()