import networks.GCN as GCN
import utils.plot_loss as plot_loss
import torch

def test_model(model, data):
    model.eval() # Set the model to evaluation mode

    # Compute the output of the model on the input data.
    out = model(data)

    # Get the predicted class by taking the argmax of the output.
    pred = out.argmax(dim=1)

    # Calculate the accuracy by comparing the predicted classes to the true labels for the test set.
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Test Accuracy: {acc:.4f}')

    return

def GCN_two_layer_training(data, num_classes, epochs=200, dataset_name=''):
    
    # instantiate the model
    # Current implementation uses 64 hidden channels, but this can be adjusted as needed.
    model = GCN.GCN_Two_Layer(data.num_features, 64, num_classes)

    # use an optimizer (e.g., Adam) to update the model parameters during training.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train() # Set the model to training mode.

    # Create a loss list to store the loss values for each epoch.
    loss_list = []

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

    plot_loss.plot_loss(loss_list, epochs, 'GCN Two Layer', dataset_name)

    # Begin testing phase
    test_model(model, data)

    return

def GCN_three_layer_training(data, num_classes, epochs=200, dataset_name=''):
    # instantiate the model
    # Current implementation uses 64 hidden channels for the first two layers, and the final layer outputs the number of classes.
    model = GCN.GCN_Three_Layer(data.num_features, 64, num_classes)

    # use an optimizer (e.g., Adam) to update the model parameters during training.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train() # Set the model to training mode.

    # Create a loss list to store the loss values for each epoch.
    loss_list = []

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

    plot_loss.plot_loss(loss_list, epochs, 'GCN Three Layer', dataset_name)

    # Begin testing phase
    test_model(model, data)

    return