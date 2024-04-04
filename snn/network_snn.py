import numpy as np
import snntorch as snn
import torch
import torch.nn as nn


class SpikingNN(nn.Module):
    def __init__(self, beta, T, n_input, n_hidden, n_output):
        super(SpikingNN, self).__init__()

        self.beta = beta
        self.T = T

        self.fc1 = nn.Linear(n_input, n_hidden[0])
        self.lif1 = snn.Leaky(beta=self.beta)
        self.fc2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.lif2 = snn.Leaky(beta=self.beta)
        self.fc3 = nn.Linear(n_hidden[1], n_output)
        self.lif3 = snn.Leaky(beta=self.beta, reset_mechanism='none')

    def forward(self, obs):
        # initialize the states of spiking neurons
        lif1_mem = self.lif1.init_leaky()
        lif2_mem = self.lif2.init_leaky()
        lif3_mem = self.lif3.init_leaky()

        # record outputs
        lif1_spk_rec = []
        lif2_spk_rec = []
        lif3_mem_rec = []

        for step in range(self.T):
            cur1 = self.fc1(obs)
            lif1_spk, lif1_mem = self.lif1(cur1, lif1_mem)
            cur2 = self.fc2(lif1_spk)
            lif2_spk, lif2_mem = self.lif2(cur2, lif2_mem)
            cur3 = self.fc3(lif2_spk)
            _, lif3_mem = self.lif3(cur3, lif3_mem)

            lif1_spk_rec.append(lif1_spk)
            lif2_spk_rec.append(lif2_spk)
            lif3_mem_rec.append(lif3_mem)

        # convert lists to tensors
        lif1_spk_rec = torch.stack(lif1_spk_rec)
        lif2_spk_rec = torch.stack(lif2_spk_rec)
        lif3_mem_rec = torch.stack(lif3_mem_rec)

        return lif1_spk_rec, lif2_spk_rec, lif3_mem_rec

