import torch
from torch import nn
# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import argparse
from LinearRegressionModel_ch01 import LinearRegressionModel
from LinearRegressionModelV2_ch01 import LinearRegressionModelV2
from pathlib import Path

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

torch.manual_seed(42)
model_1 = LinearRegressionModelV2()


# print(list(model_0.parameters()))
# print(model_0.state_dict())

with torch.inference_mode():
    y_preds = model_0(X_test)
    yV2_preds = model_1(X_test)

# print(f"y_preds = {y_preds}")

epoch_count = []
loss_values = []
test_loss_values = []

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "default.pth"
MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

def trainAndEvalModel0() -> torch.Tensor:
    # Setup the loss function
    loss_fn = nn.L1Loss()

    # Setup the optimizer (Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.0001)
    epochs = 25000
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

        # Set the model to testing mode
        model_0.eval()  # Turns off gradient tracking
        with torch.inference_mode():
            # Do the forward pass
            y_preds_new = model_0(X_test)

            # Calculate the test loss
            test_loss = loss_fn(y_preds_new, y_test)

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            loss_values.append(loss.to("cpu").detach().numpy())
            test_loss_values.append(test_loss.to("cpu").detach().numpy())
            print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
            print(model_0.state_dict())
    return y_preds_new

def trainAndEvalModel1() -> torch.Tensor:
    model_1 = LinearRegressionModelV2()
    loss_fn = nn.L1Loss()
    lossV2Temp = 0
    optimizer_v2 = torch.optim.SGD(params=model_1.parameters(), lr=0.0001)
    torch.manual_seed(42)
    epochs = 44000
    for epoch in range(epochs):
        model_1.train()
        
        yV2_preds = model_1(X_train)
        
        lossV2 = loss_fn(yV2_preds, y_train)
        
        optimizer_v2.zero_grad()
        
        lossV2.backward()
        
        optimizer_v2.step()
        
        model_1.eval()
        with torch.inference_mode():
            yV2_preds_new = model_1(X_test)
            test_lossV2 = loss_fn(yV2_preds_new, y_test)
            
        if epoch %10 == 0:
            print(f"Epoch: {epoch} | Loss: {lossV2} | Test loss: {test_lossV2}")
            print(model_1.state_dict())
            if lossV2 != lossV2Temp:
                lossV2Temp = lossV2
            else: 
                break
    
    
    MODEL_NAME = "01_workflow_model_1.pth"
    print(f"Saving model to path: {MODEL_PATH/MODEL_NAME}")
    torch.save(obj=model_1.state_dict(), f=MODEL_PATH/MODEL_NAME)
            
    return yV2_preds_new
        
        

# trainAndEvalModel0()
# trainAndEvalModel1()
# Create models directory
# MODEL_PATH = Path("models")
# MODEL_PATH.mkdir(parents=True, exist_ok=True)

# # Create model save path
# MODEL_NAME = "01_workflow_model_0.pth"
# MODEL_SAVE_PATH = MODEL_PATH/MODEL_NAME

# # Save the model state_dict
# # print(f"Saving model to path: {MODEL_SAVE_PATH}")
# # torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

# # loading a PyTorch Model
# # To load in a saved state_dict we have to instantiate a new instance of our model class
# loaded_model_0 = LinearRegressionModel()
# print(f"Loaded Model 0 State Dict = {loaded_model_0.state_dict()}")
# # Load the saved state_dict of model_0
# loaded_model_0.load_state_dict(torch.load(MODEL_SAVE_PATH))
# print(f"Loaded Model 0 State Dict = {loaded_model_0.state_dict()}")

# loaded_model_0.eval()
# with torch.inference_mode():
#     y_preds = model_0(X_test)
#     loaded_model_preds = loaded_model_0(X_test)

# print(y_preds == loaded_model_preds)


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
    # plot_predictions(predictions=y_preds_new.to("cpu").clone().detach())
    
    # plot_predictions(predictions=trainAndEvalModel1().to("cpu").clone().detach())
    
    # loading a PyTorch Model
    # To load in a saved state_dict we have to instantiate a new instance of our model class
    loaded_model_1 = LinearRegressionModelV2()
    # Load the saved state_dict of model_1
    MODEL_NAME = "01_workflow_model_1.pth"
    loaded_model_1.load_state_dict(torch.load(MODEL_PATH/MODEL_NAME))
    print(f"Loaded Model 1 State Dict = {loaded_model_1.state_dict()}")
    loaded_model_1.eval()
    with torch.inference_mode():
        loaded_model_1_y_preds = loaded_model_1(X_test)
    
    plot_predictions(predictions=loaded_model_1_y_preds.to("cpu").clone().detach())


    # plt.plot(epoch_count, loss_values, label="Train loss")
    # plt.plot(epoch_count, test_loss_values, label="Test loss")
    # plt.title("Training and test loss curves")
    # plt.ylabel("Loss")
    # plt.xlabel("Epochs")
    # plt.show()
    # print()
