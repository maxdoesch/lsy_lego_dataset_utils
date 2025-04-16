import os
import argparse
import random
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import utils

class DatasetVisualizer:
    def __init__(self, args):

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
        self.modality = args.modality
        self.type = args.type
        self.gripper = args.gripper

        self.slider_val = 0
    
    def vis_episode(self):
        trajectory_path = os.path.join(self.episode_path, 'trajectory.h5')
        if not os.path.exists(trajectory_path):
            raise FileNotFoundError(f"Trajectory file missing: {trajectory_path}")
        
        with h5py.File(trajectory_path, 'r') as episode:
            steps = [f'step_{i}' for i in range(len(episode)) if f'step_{i}' in episode]
            if not steps:
                raise ValueError("No steps found in episode data.")
            
            third_person_images = [episode[step]['observation']['third_person_image'][()] for step in steps]
            wrist_images = [episode[step]['observation']['wrist_image'][()] for step in steps]
            actions_cartesian = np.array([episode[step]['action']['cartesian_position_delta'][()] for step in steps])
            states_cartesian = np.array([episode[step]['observation']['cartesian_position'][()] for step in steps])
            actions_joint = np.array([episode[step]['action']['joint_position_delta'][()] for step in steps])
            states_joint = np.array([episode[step]['observation']['joint_position'][()] for step in steps])
            gripper_position = np.array([episode[step]['observation']['gripper_position'][()] for step in steps])
            if 'gripper' in episode[steps[0]]['action']:
                gripper_action = np.array([episode[step]['action']['gripper'][()] for step in steps])
            else:
                gripper_action = None
            caption = str(episode[steps[-1]]['instruction'][()].decode())
        
        third_person_strip = np.stack(third_person_images, axis=0)
        wrist_strip = np.stack(wrist_images, axis=0)
        third_person_strip, wrist_strip = utils.pad_to_match(third_person_strip, wrist_strip)
        combined_strip = np.concatenate((third_person_strip, wrist_strip), axis=2)
        
        num_plots = 3 if self.modality == 'both' else 2
        fig, axs = plt.subplots(num_plots, 1, figsize=(20, 12))
        fig.suptitle(f"Language Instruction: {caption}, {steps[0]}")
        axs = [axs] if num_plots == 1 else axs

        plot_titles = {
            'state': 'State',
            'action': 'Action',
            'both': 'State and Action'
        }
        plot_title = plot_titles.get(self.type, 'Unknown')

        axs[0].imshow(combined_strip[self.slider_val])
        axs[0].axis('off')
        
        if self.modality in ['cartesian', 'both']:
            if self.type in ['state', 'both']:
                cartesian_state_labels = [r'$x$', r'$y$', r'$z$', r'$\theta_1$', r'$\theta_2$', r'$\theta_3$']
                axs[1].plot(states_cartesian, label=cartesian_state_labels)

            if self.type in ['action', 'both']:
                cartesian_action_labels = [r'$\Delta x$', r'$\Delta y$', r'$\Delta z$', r'$\Delta \theta_1$', r'$\Delta \theta_2$', r'$\Delta \theta_3$']
                axs[1].plot(actions_cartesian, linestyle='dashed', label=cartesian_action_labels)

            if self.gripper in ['action', 'both'] and gripper_action is not None:
                axs[1].step(range(len(gripper_action)), gripper_action * np.max(actions_cartesian), where='pre', label="Gripper")
            if self.gripper in ['state', 'both']:
                axs[1].plot(gripper_position * np.max(actions_cartesian) / np.max(gripper_position), label="Gripper")

            state_line = axs[1].axvline(self.slider_val, color='r', linestyle='--')
            axs[1].set_title(f"Cartesian {plot_title}")
            axs[1].legend(fontsize='small')
            axs[1].grid()

        
        if self.modality in ['joint', 'both']:
            idx = 2 if self.modality == 'both' else 1

            if self.type in ['state', 'both']:
                joint_state_labels = [fr'$\theta_{i}$' for i in range(1, 8)]
                axs[idx].plot(states_joint, label=joint_state_labels)

            if self.type in ['action', 'both']:
                joint_action_labels = [fr'$\Delta \theta_{i}$' for i in range(1, 8)]
                axs[idx].plot(actions_joint, linestyle='dashed', label=joint_action_labels)

            if self.gripper in ['action', 'both'] and gripper_action is not None:
                axs[idx].step(range(len(gripper_action)), gripper_action * np.max(actions_joint), where='pre', label="Gripper")
            if self.gripper in ['state', 'both']:
                axs[idx].plot(gripper_position * np.max(actions_joint) / np.max(gripper_position), label="Gripper")

            joint_line = axs[idx].axvline(self.slider_val, color='r', linestyle='--')
            axs[idx].set_title(f"Joint {plot_title}")
            axs[idx].legend(fontsize='small')
            axs[idx].grid()
        
        ax_slider = plt.axes([0.25, 0.05, 0.65, 0.03])
        slider = Slider(ax_slider, 'Timestep', 0, len(steps)-1, valinit=self.slider_val, valfmt='%0.0f')
        
        def update(val):
            idx = int(slider.val)

            fig.suptitle(f"Language Instruction: {caption}, {steps[idx]}")

            axs[0].imshow(combined_strip[idx])

            if self.modality in ['cartesian', 'both']:
                state_line.set_xdata([idx])
            if self.modality in ['joint', 'both']:
                joint_line.set_xdata([idx])
            
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize a given episode of the Lego Dataset.")
    parser.add_argument("--episode", type=int, required=False, help="ID of the episode to visualize. If not provided, a random episode is selected.")
    parser.add_argument("--modality", default='both', choices=['cartesian', 'joint', 'both'], help="Visualize joint, cartesian, or both.")
    parser.add_argument("--type", default='both', choices=['both', 'state', 'action'], help="Visualize states, actions, or both.")
    parser.add_argument("--gripper", default='action', choices=['state', 'action', 'both', 'none'] ,help="Visualize the gripper.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset directory.")
    
    args = parser.parse_args()
    visualizer = DatasetVisualizer(args)
    visualizer.vis_episode()

if __name__ == "__main__":
    main()
