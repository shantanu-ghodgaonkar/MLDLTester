import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse
from circleModelv0_ch02 import CircleModelV0

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

# number of samples
n_samples = 1000

# create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# make DataFrame of circle
circles = pd.DataFrame({"X1":X[:,0], 
                        "X2":X[:,1],
                        "label":y})

# plt.scatter(x=X[:,0], y=X[:,1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

X_sample = X[0]
y_sample = y[0]

# print(f"Values for one sample of X: {X_sample} and the same for y: {y_sample}")
# print(f"Shape for one sampel for X: {X_sample.shape} and the same for y: {y_sample.shape}")

# Turn data into tensors 
X = torch.from_numpy(X).type(torch.float32).to(args.device)
y = torch.from_numpy(y).type(torch.float32).to(args.device)

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_0 = CircleModelV0().to(args.device)

if __name__ == '__main__':
    print("DEBUG POINT")
