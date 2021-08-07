import torch
import numpy as np
import pandas as pd
import scipy.stats as stats


def generate_block_data(num_of_process=6, num_of_blocks=50, size=1, distribution="lognormal"):

    if distribution == "lognormal":
        shape = [0.543, 0.525, 0.196, 0.451, 0.581, 0.432]
        scale = [2.18, 2.18, 0.518, 2.06, 1.79, 2.10]
    elif distribution == "uniform":
        loc = [0 for _ in range(num_of_process)]
        scale = [10 for _ in range(num_of_process)]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    process_time_temp = np.zeros((size * num_of_blocks, num_of_process))
    process_time = {}

    for i in range(num_of_process):
        if distribution == "lognormal":
            r = np.round(stats.lognorm.rvs(shape[i], loc=0, scale=scale[i], size=size * num_of_blocks), 1)
        elif distribution == "uniform":
            r = np.round(stats.uniform.rvs(loc=loc[i], scale=scale[i],size=size * num_of_blocks), 1)
        process_time_temp[:, i] = r
    process_time_temp = process_time_temp.reshape((size, num_of_blocks, num_of_process))

    for i in range(size):
        process_time[str(i)] = torch.FloatTensor(process_time_temp[i]).to(device)

    return process_time


def read_block_data(filepath):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    process_time = {}

    df_process_time = pd.read_excel(filepath, sheet_name=None)
    for key, value in df_process_time.items():
        process_time[key] = torch.FloatTensor(process_time).to(device)

    return process_time


if __name__ == "__main__":
    blocks = generate_block_data()
    print("f")