import numpy as np
import matplotlib.pyplot as plt

# Inputs:
# - cm: The confusion matrix to be plotted, typically a 2D array where cm[i, j] represents the number of instances of class i that were predicted as class j.
# - classes: A list of class labels corresponding to the indices of the confusion matrix, used for labeling the axes of the plot.
# - title: The title for the plot, indicating the model and configuration.
# - dataset_name: The name of the dataset used for training, to be included in the plot title and filename.
def plot_confusion_matrix(cm, classes, title, dataset_name):
    plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{title} - {dataset_name}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'figures/{title.replace(" ", "_")}_{dataset_name}_confusion_matrix.png')

    return
