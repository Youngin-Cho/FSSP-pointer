import torch
import numpy as np


class PanelBlockShop:
    def __init__(self, num_of_process=6, num_of_blocks=50):
        self.num_of_process = num_of_process
        self.num_of_blocks = num_of_blocks

    def calculate_makespan(self, blocks, sequence):
        if isinstance(blocks, torch.Tensor):
            blocks_numpy = blocks.cpu().numpy()
        else:
            blocks_numpy = blocks

        if isinstance(sequence, torch.Tensor):
            sequence_numpy = sequence.cpu().numpy()
        else:
            sequence_numpy = sequence

        temp = np.zeros((self.num_of_blocks + 1, self.num_of_process + 1))
        for i in range(1, self.num_of_blocks + 1):
            temp[i, 0] = 0
            for j in range(1, self.num_of_process + 1):
                if i == 1:
                    temp[0, j] = 0

                if temp[i - 1, j] > temp[i, j - 1]:
                    temp[i, j] = temp[i - 1, j] + blocks_numpy[sequence_numpy[i - 1], j - 1]
                else:
                    temp[i, j] = temp[i, j - 1] + blocks_numpy[sequence_numpy[i - 1], j - 1]
        C_max = temp[self.num_of_blocks, self.num_of_process]

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        return torch.FloatTensor([C_max]).to(device)


