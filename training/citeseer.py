# Script dedicated to importing and pre-processing the citeseer dataset, which is a common benchmark for graph neural networks.
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

# Function to load the citeseer dataset and return the graph data and number of classes.
def load_citeseer():
    # Load the citeseer dataset using PyTorch Geometric's Planetoid class.
    dataset = Planetoid(root='data/Citeseer', name='Citeseer', transform=NormalizeFeatures())
    data = dataset[0]  # Get the graph data object from the dataset.
    
    return data, dataset.num_classes  # Return the graph data and the number of classes for classification.