import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, ouput_dim=10):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ouput_dim)
        )
    
    def forward(self, x):
        return self.layers(x)