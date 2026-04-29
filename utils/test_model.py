import sklearn.metrics as metrics
import utils.plot_confusion_matrix as plot_confusion_matrix

# Define a function to test the model on the test set and evaluate its performance.
def test_model(model, data, model_name='', dataset_name=''):
    model.eval() # Set the model to evaluation mode

    # Compute the output of the model on the input data.
    out = model(data)

    # Get the predicted class by taking the argmax of the output.
    pred = out.argmax(dim=1)

    # Calculate the accuracy by comparing the predicted classes to the true labels for the test set.
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Test Accuracy: {acc:.4f}')\
    
    # Calculate the confusion matrix for the test set.
    cm = metrics.confusion_matrix(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
    print(f'Confusion Matrix:\n{cm}')

    # Plot the confusion matrix using the utility function.
    plot_confusion_matrix.plot_confusion_matrix(cm, classes=range(data.y.max().item() + 1), title=f'{model_name} Confusion Matrix', dataset_name=dataset_name)

    # Report the classification metrics (precision, recall, f1-score) for the test set.
    report = metrics.classification_report(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())
    print(f'Classification Report:\n{report}')

    # Write the report to a text file for record-keeping.
    with open(f'accuracy_reports/{model_name.replace(" ", "_")}_{dataset_name}_classification_report.txt', 'w') as f:
        f.write(report)

    return