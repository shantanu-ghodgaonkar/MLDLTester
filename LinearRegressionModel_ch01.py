import torch
from torch import nn
import argparse


class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
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
        self.weight = nn.Parameter(torch.randn(
            1, requires_grad=True, dtype=torch.float, device=args.device))
        self.bias = nn.Parameter(torch.randn(
            1, requires_grad=True, dtype=torch.float, device=args.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


if __name__ == '__main__':
    print("This class contains the Linear Regression Model developed in chapter 01 of the course")
