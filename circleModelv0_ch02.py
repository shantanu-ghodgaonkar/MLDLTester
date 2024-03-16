import torch
from torch import nn


class CircleModelV0(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # create two nn.Linear layers capable of handling our data
        self.layer_1 = nn.Linear(in_features=2, out_features=5) # takes in 2 features and upscales to 5 features
        self.layer_2 = nn.Linear(in_features=5, out_features=1) # takes in 5 features from previous layer and outputs a single feature
    
    # define a forward method to define the forward pass
    def forward(self, x):
        return self.layer_2(self.layer_1(x))