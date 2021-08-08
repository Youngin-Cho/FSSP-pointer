import torch
import random
import numpy as np
import pandas as pd

from time import time
from datetime import datetime

from environment.env import PanelBlockShop
from environment.panelblock import *
from agent.search import *


def test_model(env, params, data, makespan_path=None, time_path=None):
    date = datetime.now().strftime('%m%d_%H_%M')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    param_path = params["log_dir"] + '/%s_%s_param.csv' % (date, "train")
    print(f'generate {param_path}')
    with open(param_path, 'w') as f:
        f.write(''.join('%s,%s\n' % item for item in params.items()))

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

        print("SPT ...")
        t1 = time()
        sequence_spt, makespan_spt = random_sequence(env, process_time)
        t2 = time()
        t_spt = t2 - t1

        print("LPT ...")
        t1 = time()
        sequence_lpt, makespan_lpt = random_sequence(env, process_time)
        t2 = time()
        t_lpt = t2 - t1

        if makespan_path is None:
            makespan_path = params["test_dir"] + '/%s_makespan.csv' % date
            with open(makespan_path, 'w') as f:
                f.write('RL,Palmer,Campbell,RANDOM,SPT,LPT\n')
        else:
            with open(makespan_path, 'a') as f:
                f.write('%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,\n' % (makespan_rl, makespan_palmer, makespan_campbell,
                                                              makespan_random, makespan_spt, makespan_lpt))

        if time_path is None:
            time_path = params["test_dir"] + '/%s_time.csv' % date
            with open(time_path, 'w') as f:
                f.write('RL,Palmer,Campbell,RANDOM,SPT,LPT\n')
        else:
            with open(time_path, 'a') as f:
                f.write('%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,\n' % (t_rl, t_palmer, t_campbell, t_random, t_spt, t_lpt))



if __name__ == '__main__':

    model_path = "./result/model/"
    data_path = "../data/"

    log_dir = "./result/log/"
    test_dir = "./result/test/"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    params = {
        "num_of_process": 6,
        "num_of_blocks": 40,
        "model_path": model_path,
        "log_dir": log_dir,
        "test_dir": test_dir,
        "batch_size": 20,
        "clip_logits": 1,
        "softmax_T": 1.5,
        "decode_type": "greedy",
        "n_glimpse": 1,
    }

    env = PanelBlockShop(params["num_of_process"], params["num_of_blocks"])
    data = read_block_data(data_path)
    test_model(env, params, data)