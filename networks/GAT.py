import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv

# Define a Graph Attention Network (GAT) class with two layers.
class GAT_Two_Layer(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads=4):
        super(GAT_Two_Layer, self).__init__()
        # First GAT layer: multi-head attention with concatenation.
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, concat=True)
        # Second GAT layer: single-head for final classification.
        self.conv2 = GATConv(hidden_channels * heads, num_classes, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # Get node features and edge indices.
        
        x = self.conv1(x, edge_index)  # Apply first GAT layer.
        x = F.relu(x)  # ReLU activation.
        x = F.dropout(x, training=self.training)  # Dropout.
        
        x = self.conv2(x, edge_index)  # Apply second GAT layer.
        
        return F.log_softmax(x, dim=1)  # Log probabilities.

# Define a Graph Attention Network (GAT) class with three layers.
class GAT_Three_Layer(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads=4):
        super(GAT_Three_Layer, self).__init__()
        # First GAT layer: multi-head attention with concatenation.
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, concat=True)
        # Second GAT layer: multi-head attention with concatenation.
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
        # Third GAT layer: single-head for final classification.
        self.conv3 = GATConv(hidden_channels * heads, num_classes, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index  # Get node features and edge indices.
        
        x = self.conv1(x, edge_index)  # Apply first GAT layer.
        x = F.relu(x)  # ReLU activation.
        x = F.dropout(x, training=self.training)  # Dropout.
        
        x = self.conv2(x, edge_index)  # Apply second GAT layer.
        x = F.relu(x)  # ReLU activation.
        x = F.dropout(x, training=self.training)  # Dropout.
        
        x = self.conv3(x, edge_index)  # Apply third GAT layer.
        
        return F.log_softmax(x, dim=1)  # Log probabilities.
