import numpy as np
import matplotlib.pyplot as plt

# Inputs:
# - loss_list: A list of loss values recorded during training.
# - epochs: The total number of epochs for which the model was trained.
# - title: The title for the plot, indicating the model and configuration.
# - dataset_name: The name of the dataset used for training, to be included in the plot title and filename.
def plot_loss(loss_list, epochs, title, dataset_name):
    plt.figure(figsize=(3.5, 3))
    plt.plot(range(epochs), loss_list, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{title} - {dataset_name}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'figures/{title.replace(" ", "_")}_{dataset_name}_loss_plot.png')

    return