import torch
import random
import numpy as np
import pandas as pd

from time import time
from datetime import datetime

from environment.env import PanelBlockShop
from environment.panelblock import *
from agent.search import *


def test_model(env, params, data, test_path=None):
    date = datetime.now().strftime('%m%d_%H_%M')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    for key, process_time in data.items():

        print('pointer network ...')
        t1 = time()
        sequence_rl, makespan_rl = sampling(env, params, process_time)
        t2 = time()
        t_rl = t2 - t1

        print('Palmer heuristics ...')
        t1 = time()
        sequence_palmer, makespan_palmer = Palmer_sequence(env, process_time)
        t2 = time()
        t_palmer = t2 - t1

        print("Campbell heuristics ...")
        t1 = time()
        sequence_campbell, makespan_campbell = Campbell_sequence(env, process_time)
        t2 = time()
        t_campbell = t2 - t1

        print("Random ...")
        t1 = time()
        sequence_random, makespan_random = random_sequence(env, process_time)
        t2 = time()
        t_random = t2 - t1

        if test_path is None:
            test_path = params["test_dir"] + '/%s_results.csv' % date
            with open(test_path, 'w') as f:
                f.write('RL,Palmer,Campbell,RANDOM,SPT,LPT\n')
        else:
            with open(test_path, 'a') as f:
                f.write('%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,\n' % (C_RL, C_RANDOM, C_Palmer, C_Campbell, C_SPT, C_LPT))


if __name__ == '__main__':

    params_path = "./result/log/"
    model_path = "./result/model/"
    data_path = "../data/"
    test_path = "./result/test/"

    params = {
        "param_path": params_path,
        "model_path": model_path,
        "test_path": test_path,
        "batch_size": 20,
        "clip_logits": 1,
        "softmax_T": 1.5,
        "decode_type": "greedy",
        "n_glimpse": 1,
    }

    env = PanelBlockShop(params["num_of_process"], params["num_of_blocks"])
    data = read_block_data(data_path)
    test_model(env, params, data)