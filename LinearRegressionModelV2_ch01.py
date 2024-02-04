import torch
from torch import nn
import argparse


class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        parser = argparse.ArgumentParser(description='PyTorch Example')
        parser.add_argument(
            '--disable-cuda', action='store_true', help='Disable CUDA')
        args = parser.parse_args()
        args.device = None
        if not args.disable_cuda and torch.cuda.is_available():
            args.device = torch.device('cuda')
        else:
            args.device = torch.device('cpu')

        self.linear_layer = nn.Linear(
            in_features=1, out_features=1, device=args.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
