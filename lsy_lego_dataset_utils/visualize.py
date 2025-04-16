import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import lsy_lego_dataset_utils.utils as utils

class EpisodeVisualizer:
    def __init__(self, settings: dict):
        self.modality = settings['modality']
        self.type = settings['type']
        self.gripper = settings['gripper']

        num_plots = 3 if self.modality == 'both' else 2
        self.fig, self.axs = plt.subplots(num_plots, 1, figsize=(20, 12))
        self.axs = [self.axs] if num_plots == 1 else self.axs

        plot_titles = {
            'state': 'State',
            'action': 'Action',
            'both': 'State and Action'
        }
        self.plot_title = plot_titles.get(self.type, 'Unknown')

        self.episode_len = 0
        self.third_person_images = []
        self.wrist_images = []
        self.actions_cartesian = []
        self.states_cartesian = []
        self.actions_joint = []
        self.states_joint = []
        self.gripper_position = []
        self.gripper_action = []
        self.caption = ""
    
    def visualize_episode(self, episode: dict):
        self.episode_len = len(episode['steps'])
        self.third_person_images = [episode['steps'][step]['observation']['third_person_image'] for step in range(self.episode_len)]
        self.wrist_images = [episode['steps'][step]['observation']['wrist_image'] for step in range(self.episode_len)]
        self.actions_cartesian = np.array([episode['steps'][step]['action']['cartesian_position_delta'] for step in range(self.episode_len)])
        self.states_cartesian = np.array([episode['steps'][step]['observation']['cartesian_position'] for step in range(self.episode_len)])
        self.actions_joint = np.array([episode['steps'][step]['action']['joint_position_delta'] for step in range(self.episode_len)])
        self.states_joint = np.array([episode['steps'][step]['observation']['joint_position'] for step in range(self.episode_len)])
        self.gripper_position = np.array([episode['steps'][step]['observation']['gripper_position'] for step in range(self.episode_len)])
        if 'gripper' in episode['steps'][0]['action']:
            self.gripper_action = np.array([episode['steps'][step]['action']['gripper'] for step in range(self.episode_len)])
        else:
            self.gripper_action = None

        slider_val = 0

        self.caption = str(episode['steps'][slider_val]['instruction'].decode())

        self.fig.suptitle(f"Language Instruction: {self.caption}, {slider_val}")
        self._make_image_plot(slider_val, 0)

        if self.modality in ['cartesian', 'both']:
            self._make_cartesian_plot(1)
            state_line = self._make_vertical_line(slider_val, 1)
        
        if self.modality in ['joint', 'both']:
            idx = 2 if self.modality == 'both' else 1
            self._make_joint_plot(idx)
            joint_line = self._make_vertical_line(slider_val, idx)

        ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
        slider = Slider(ax_slider, 'Timestep', 0, self.episode_len - 1, valinit=slider_val, valfmt='%0.0f')
        
        def update(val):
            idx = int(slider.val)

            self.fig.suptitle(f"Language Instruction: {self.caption}, {idx}")

            self._make_image_plot(idx, 0)

            if self.modality in ['cartesian', 'both']:
                state_line.set_xdata([idx])
            if self.modality in ['joint', 'both']:
                joint_line.set_xdata([idx])
            
            self.fig.canvas.draw_idle()
        
        slider.on_changed(update)
        plt.show()

    def visualize_episode_live(self, step: dict):
        self.episode_len += 1
        self.third_person_images.append(step['observation']['third_person_image'])
        self.wrist_images.append(step['observation']['wrist_image'])
        self.actions_cartesian.append(step['action']['cartesian_position_delta'])
        self.states_cartesian.append(step['observation']['cartesian_position'])
        self.actions_joint.append(step['action']['joint_position_delta'])
        self.states_joint.append(step['observation']['joint_position'])
        self.gripper_position.append(step['observation']['gripper_position'])
        if 'gripper' in step['action']:
            self.gripper_action.append(step['action']['gripper'])
        else:
            self.gripper_action = None
        self.caption = step['instruction']

        self._update_plot(self.episode_len - 1)

    def _make_image_plot(self, time_step: int, axis_idx: int):
        assert time_step < self.episode_len, f"Time step {time_step} exceeds episode length {self.episode_len}."

        padded_third_person_image, padded_wrist_image = utils.pad_to_match(self.third_person_images[time_step], self.wrist_images[time_step])
        combined_image = np.concatenate((padded_third_person_image, padded_wrist_image), axis=1)

        self.axs[axis_idx].imshow(combined_image)
        self.axs[axis_idx].axis('off')

    def _make_cartesian_plot(self, axis_idx: int):
        if self.type in ['state', 'both']:
            cartesian_state_labels = [r'$x$', r'$y$', r'$z$', r'$\theta_1$', r'$\theta_2$', r'$\theta_3$']
            self.axs[axis_idx].plot(self.states_cartesian, label=cartesian_state_labels)

        if self.type in ['action', 'both']:
            cartesian_action_labels = [r'$\Delta x$', r'$\Delta y$', r'$\Delta z$', r'$\Delta \theta_1$', r'$\Delta \theta_2$', r'$\Delta \theta_3$']
            self.axs[axis_idx].plot(self.actions_cartesian, linestyle='dashed', label=cartesian_action_labels)

        if self.gripper in ['action', 'both'] and self.gripper_action is not None:
            self.axs[axis_idx].step(range(len(self.gripper_action)), np.array(self.gripper_action) * np.max(self.actions_cartesian), where='pre', label="Gripper")
        if self.gripper in ['state', 'both']:
            self.axs[axis_idx].plot(np.array(self.gripper_position) * np.max(self.actions_cartesian) / np.max(self.gripper_position), label="Gripper")

        self.axs[axis_idx].set_title(f"Cartesian {self.plot_title}")
        self.axs[axis_idx].legend(fontsize='small')
        self.axs[axis_idx].grid()

    def _make_joint_plot(self, axis_idx: int):
        if self.type in ['state', 'both']:
            joint_state_labels = [fr'$\theta_{i}$' for i in range(1, 8)]
            self.axs[axis_idx].plot(self.states_joint, label=joint_state_labels)

        if self.type in ['action', 'both']:
            joint_action_labels = [fr'$\Delta \theta_{i}$' for i in range(1, 8)]
            self.axs[axis_idx].plot(self.actions_joint, linestyle='dashed', label=joint_action_labels)

        if self.gripper in ['action', 'both'] and self.gripper_action is not None:
            self.axs[axis_idx].step(range(len(self.gripper_action)), np.array(self.gripper_action) * np.max(self.actions_joint), where='pre', label="Gripper")
        if self.gripper in ['state', 'both']:
            self.axs[axis_idx].plot(np.array(self.gripper_position) * np.max(self.actions_joint) / np.max(self.gripper_position), label="Gripper")

        self.axs[axis_idx].set_title(f"Joint {self.plot_title}")
        self.axs[axis_idx].legend(fontsize='small')
        self.axs[axis_idx].grid()

    def _make_vertical_line(self, time_step: int, axis_idx: int):
        return self.axs[axis_idx].axvline(time_step, color='r', linestyle='--')

    def _update_plot(self, time_step: int):
        for ax in self.axs:
            ax.clear()

        self.fig.suptitle(f"Language Instruction: {self.caption}, {self.episode_len}")
        self._make_image_plot(time_step, 0)
        if self.modality in ['cartesian', 'both']:
            self._make_cartesian_plot(1)
        
        if self.modality in ['joint', 'both']:
            idx = 2 if self.modality == 'both' else 1
            self._make_joint_plot(idx)

        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()

        plt.pause(0.01)

class DatasetVisualizer:
    def __init__(self, args):
        self.args = args
        self.episode_visualizer = EpisodeVisualizer({
            'modality': args.modality,
            'type': args.type,
            'gripper': args.gripper
        })

        if args.dataset_path is not None:
            dataset_path = args.dataset_path
        else:
            workspace_path = os.getcwd()
            dataset_path = os.path.join(workspace_path, 'dataset')
        
        episodes = [f for f in os.listdir(dataset_path) if f.startswith('episode_')]
        if not episodes:
            raise FileNotFoundError("No episodes found in dataset directory.")
        
        episode_name = f'episode_{args.episode}' if args.episode is not None else random.choice(episodes)
        if episode_name not in episodes:
            raise ValueError(f"Episode {episode_name} not found.")
        
        print(f"Visualizing {episode_name}.")
        self.episode_path = os.path.join(dataset_path, episode_name)

    def visualize_episode(self):
        trajectory_path = os.path.join(self.episode_path, 'trajectory.h5')
        if not os.path.exists(trajectory_path):
            raise FileNotFoundError(f"Trajectory file missing: {trajectory_path}")
        
        episode = utils.load_h5_file(trajectory_path)
        if not episode:
            raise ValueError("No steps found in episode data.")
        self.episode_visualizer.visualize_episode(episode)

def main():
    parser = argparse.ArgumentParser(description="Visualize a given episode of the Lego Dataset.")
    parser.add_argument("--episode", type=int, required=False, help="ID of the episode to visualize. If not provided, a random episode is selected.")
    parser.add_argument("--modality", default='both', choices=['cartesian', 'joint', 'both'], help="Visualize joint, cartesian, or both.")
    parser.add_argument("--type", default='both', choices=['both', 'state', 'action'], help="Visualize states, actions, or both.")
    parser.add_argument("--gripper", default='action', choices=['state', 'action', 'both', 'none'] ,help="Visualize the gripper.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset directory.")
    
    args = parser.parse_args()
    visualizer = DatasetVisualizer(args)
    visualizer.visualize_episode()

if __name__ == "__main__":
    main()
