import os
import numpy as np
import matplotlib.pyplot as plt


def load_model_data(filename):
    filepath = os.path.join('Q3_Images', filename)

    if os.path.isfile(filepath):
        data = np.load(filepath, allow_pickle=True).item()
        return data
    else:
        return {}

def plot_q3_r100_rewards(environment):
    # Load the data
    data = load_model_data('r100_values.npy')
    plt.figure(figsize=(10, 5))

    # Plot the data for each model
    for model, r100_values in data.items():
        plt.plot(r100_values, label=model)

    plt.title(f'{environment} R100 Rewards vs Episode Number')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)

    # Save the figure
    images_dir = 'Q3_Images'
    filename = os.path.join(images_dir, f'{environment}_reward100.png')
    plt.savefig(filename)
    plt.show()


def plot_losses(environment):
    # Load the data
    data = load_model_data('losses.npy')
    plt.figure(figsize=(10, 5))

    # Plot the data for each model
    for model, losses in data.items():
        plt.plot(losses, label=model)

    plt.title(f'{environment} Losses vs Epoch Number')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the figure
    images_dir = 'Q3_Images'
    filename = os.path.join(images_dir, f'{environment}_losses.png')
    plt.savefig(filename)
    plt.show()


plot_q3_r100_rewards('Cartpole-V0')
plot_losses('Cartpole-V0')