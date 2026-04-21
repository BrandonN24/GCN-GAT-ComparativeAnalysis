import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv

# Define a Graph Convolutional Network (GCN) class that inherits from PyTorch's nn.Module.
class GCN_Two_Layer(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN_Two_Layer, self).__init__()
        # Define the first GCN layer that takes in the number of features and outputs hidden channels.
        self.conv1 = GCNConv(num_features, hidden_channels)
        # Define the second GCN layer that takes in hidden channels and outputs the number of classes.
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # Get node features and edge indices from the input data.
        
        x = self.conv1(x, edge_index)  # Apply the first GCN layer.
        x = F.relu(x)  # Apply ReLU activation function.
        x = F.dropout(x, training=self.training)  # Apply dropout for regularization during training.
        x = self.conv2(x, edge_index)  # Apply the second GCN layer.
        
        return F.log_softmax(x, dim=1)  # Return the log probabilities for each class.

class GCN_Three_Layer(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN_Three_Layer, self).__init__()
        # Define the first GCN layer that takes in the number of features and outputs hidden channels.
        self.conv1 = GCNConv(num_features, hidden_channels)
        # Define the second GCN layer that takes in hidden channels and outputs hidden channels.
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # Define the third GCN layer that takes in hidden channels and outputs the number of classes.
        self.conv3 = GCNConv(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # Get node features and edge indices from the input data.
        
        x = self.conv1(x, edge_index)  # Apply the first GCN layer.
        x = F.relu(x)  # Apply ReLU activation function.
        x = F.dropout(x, training=self.training)  # Apply dropout for regularization during training.
        
        x = self.conv2(x, edge_index)  # Apply the second GCN layer.
        x = F.relu(x)  # Apply ReLU activation function.
        x = F.dropout(x, training=self.training)  # Apply dropout for regularization during training.
        
        x = self.conv3(x, edge_index)  # Apply the third GCN layer.
        
        return F.log_softmax(x, dim=1)  # Return the log probabilities for each class.