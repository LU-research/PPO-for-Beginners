import snntorch as snn
import torch.nn as nn


class SpikingNN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()

        self.fc1 = nn.Linear(n_input, 64)