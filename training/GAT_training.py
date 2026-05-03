import networks.GAT as GAT
from utils import plot_acc, test_model
import utils.plot_loss as plot_loss
import torch

def GAT_two_layer_training(data, num_classes, epochs=200, dataset_name=''):
    
    # instantiate the model
    # Current implementation uses 64 hidden channels, but this can be adjusted as needed.
    model = GAT.GAT_Two_Layer(data.num_features, 64, num_classes)

    # use an optimizer (e.g., Adam) to update the model parameters during training.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train() # Set the model to training mode.

    # Create a loss list to store the loss values for each epoch.
    loss_list = []
    acc_list = []

    # Training the model for a specified number of epochs (e.g., 200).
    for epoch in range(epochs):
        # Zero the gradients.
        model.zero_grad()
        
        # Forward pass: compute the output of the model on the input data.
        out = model(data)
        
        # Compute the loss using negative log likelihood loss function.
        loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
        print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}', end='\r')
        
        # Backward pass: compute the gradients of the loss with respect to the model parameters.
        loss.backward()
        
        # Update the model parameters using an optimizer
        optimizer.step()

        # Append the loss value to the loss list for tracking.
        loss_list.append(loss.item())

        # Append the accuracy value to the accuracy list for tracking.
        pred = out.argmax(dim=1)
        correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
        acc = int(correct) / int(data.train_mask.sum())
        acc_list.append(acc)

    # Plot the loss and accuracy curves for the training process.
    plot_loss.plot_loss(loss_list, epochs, 'GAT Two Layer', dataset_name)
    plot_acc.plot_acc(acc_list, epochs, 'GAT Two Layer', dataset_name)

    # Begin testing phase
    test_model.test_model(model, data, model_name='GAT Two Layer', dataset_name=dataset_name)

    return

def GAT_three_layer_training(data, num_classes, epochs=200, dataset_name=''):
    # instantiate the model
    # Current implementation uses 64 hidden channels for the first two layers, and the final layer outputs the number of classes.
    model = GAT.GAT_Three_Layer(data.num_features, 64, num_classes)

    # use an optimizer (e.g., Adam) to update the model parameters during training.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train() # Set the model to training mode.

    # Create a loss list to store the loss values for each epoch.
    loss_list = []
    # Create an accuracy list to store the accuracy values for each epoch.
    acc_list = []

    # Training the model for a specified number of epochs (e.g., 200).
    for epoch in range(epochs):
        # Zero the gradients.
        model.zero_grad()

        # Forward pass: compute the output of the model on the input data.
        out = model(data)

        # Compute the loss using negative log likelihood loss function.
        loss = torch.nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
        print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}', end='\r')

        # Backward pass: compute the gradients of the loss with respect to the model parameters.
        loss.backward()

        # Update the model parameters using an optimizer
        optimizer.step()

        # Append the loss value to the loss list for tracking.
        loss_list.append(loss.item())
        # Append the accuracy value to the accuracy list for tracking.
        pred = out.argmax(dim=1)
        correct = (pred[data.train_mask] == data.y[data.train_mask]).sum()
        acc = int(correct) / int(data.train_mask.sum())
        acc_list.append(acc)

    # Plot the loss and accuracy curves for the training process.
    plot_loss.plot_loss(loss_list, epochs, 'GAT Three Layer', dataset_name)
    plot_acc.plot_acc(acc_list, epochs, 'GAT Three Layer', dataset_name)

    # Begin testing phase
    test_model.test_model(model, data, model_name='GAT Three Layer', dataset_name=dataset_name)

    return