import numpy as np
import pandas as pd
import scipy.stats as stats


process_num = 6
shape = [0.543, 0.525, 0.196, 0.451, 0.581, 0.432]
scale = [2.18, 2.18, 0.518, 2.06, 1.79, 2.10]


def generate_block_data(block_num=50, batch_size=1):
    process_time = np.zeros((batch_size * block_num, process_num))

    for i in range(process_num):
        r = np.round(stats.lognorm.rvs(shape[i], loc=0, scale=scale[i], size=batch_size * block_num), 1)
        process_time[:, i] = r
    process_time = process_time.reshape((batch_size, block_num, process_num))

    return process_time


def read_block_data(filepath):
    pass


if __name__ == "__main__":
    blocks = generate_block_data()
    print("f")