import torch
from torch import nn
# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import argparse
# from 01_LinearRegressionModel import LinearRegressionModel
from LinearRegressionModel_ch01 import LinearRegressionModel

# print(torch.__version__)
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

# NOTE: Linear Regression formula : Y = a + bX
weight = 0.7  # b
bias = 0.3  # a

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
X = X.to(args.device)
y = weight * X + bias

# print(f"X[:10] = {X[:10]}")
# print(f"len(X) = {len(X)}")
# print(f"y[:10] = {y[:10]}")
# print(f"len(y) = {len(y)}")

train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]

# print(f"X_train = {X_train}")
# print(f"len(X_train) = {len(X_train)}")
# print(f"y_train = {y_train}")
# print(f"len(Y_train) = {len(y_train)}")

X_test, y_test = X[train_split:], y[train_split:]

# print(f"X_test = {X_test}")
# print(f"len(X_test) = {len(X_test)}")
# print(f"y_test = {y_test}")
# print(f"len(Y_test) = {len(y_test)}")

torch.manual_seed(42)
model_0 = LinearRegressionModel()
# print(list(model_0.parameters()))
# print(model_0.state_dict())

with torch.inference_mode():
    y_preds = model_0(X_test)

# print(f"y_preds = {y_preds}")

# Setup the loss function
loss_fn = nn.L1Loss()

# Setup the optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.0001)

# An epoch is one loop through the data
epochs = 35000

# Loop through the data
for epoch in range(epochs):
    
    # Set the model to training mode
    model_0.train()  # Train mode in PyTorch that sets all parameters that need gradients to have gradients

    # Forward pass
    y_preds = model_0(X_train)
    
    # Calculate the loss 
    loss = loss_fn(y_preds, y_train)
    
    # Optimiser zero grad
    optimizer.zero_grad()
    
    # Perform backpropagation on the loss wrt the parameters of the model
    loss.backward()
    
    # Step the optimiser (Perform gradient descent)
    optimizer.step()
    
    model_0.eval()  # Turns off gradient tracking
    # if epoch == 35000:
    #     print("DEBUG POINT")

    with torch.inference_mode():
        y_preds_new = model_0(X_test)

def plot_predictions(train_data=X_train.to("cpu"), train_labels=y_train.to("cpu"), test_data=X_test.to("cpu"), test_labels=y_test.to("cpu"), predictions=None):
    """
    Plots training data, test data and compares predictions

    Args:
        train_data (_type_, optional): Training data/features. Defaults to X_train.
        train_labels (_type_, optional): Training labels. Defaults to y_train.
        test_data (_type_, optional): Test data/features. Defaults to X_test.
        test_labels (_type_, optional): Test labels. Defaults to y_test.
        predictions (_type_, optional): Predcitons. Defaults to None.
    """
    # Create a figure
    plt.figure(figsize=(10, 7))
    # Plot training data
    plt.scatter(train_data, train_labels, c='b', s=4, label="Training Data")
    # Plot testing data
    plt.scatter(test_data, test_labels, c='g', s=4, label="Testing Data")
    # Are there predictions?
    if predictions is not None:
        # Plot predictions
        plt.scatter(test_data, predictions, c='r', s=4, label="Predictions")

    # Show legend
    plt.legend(prop={"size": 14})
    # print("DEBUG POINT")
    plt.show()


if __name__ == '__main__':
    # plot_predictions(X_train, y_train, X_test, y_test, None)
    # plot_predictions(predictions=y_preds.to("cpu").clone().detach())
    plot_predictions(predictions=y_preds_new.to("cpu").clone().detach())
    # print()
