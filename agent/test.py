import torch
import random
import numpy as np
import pandas as pd

from time import time
from datetime import datetime

from environment.env import PanelBlockShop
from environment.panelblock import *
from agent.search import *
from benchmark.heuristics import *


def test_model(env, params, data, makespan_path=None, time_path=None):
    date = datetime.now().strftime('%m%d_%H_%M')
    # device = torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    param_path = params["log_dir"] + '/' + params["model"] + '/%s_%s_param.csv' % (date, "test")
    print(f'generate {param_path}')
    with open(param_path, 'w') as f:
        f.write(''.join('%s,%s\n' % item for item in params.items()))

    if makespan_path is None:
        makespan_path = params["test_dir"] + '/' + params["model"] + '/%s_makespan.csv' % date
        with open(makespan_path, 'w') as f:
            f.write('RL,NEH,Palmer,Campbell,RANDOM,SPT,LPT\n')

    if time_path is None:
        time_path = params["test_dir"] + '/' + params["model"] + '/%s_time.csv' % date
        with open(time_path, 'w') as f:
            f.write('RL,NEH,Palmer,Campbell,RANDOM,SPT,LPT\n')

    iteration = 1
    for key, process_time in data.items():
        print("=" * 50 + "iteration %d" % iteration + "=" * 50)

        print('pointer network ...')
        t1 = time()
        sequence_rl, makespan_rl = sampling(env, params, process_time)
        t2 = time()
        t_rl = t2 - t1

        print('NEH heuristics ...')
        t1 = time()
        sequence_neh, makespan_neh = NEH_sequence(env, process_time)
        t2 = time()
        t_neh = t2 - t1

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
        sequence_spt, makespan_spt = SPT_sequence(env, process_time)
        t2 = time()
        t_spt = t2 - t1

        print("LPT ...")
        t1 = time()
        sequence_lpt, makespan_lpt = LPT_sequence(env, process_time)
        t2 = time()
        t_lpt = t2 - t1

        with open(makespan_path, 'a') as f:
            f.write('%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,\n' %
                    (makespan_rl, makespan_neh, makespan_palmer, makespan_campbell,
                     makespan_random, makespan_spt, makespan_lpt))
        with open(time_path, 'a') as f:
            f.write('%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,\n' %
                    (t_rl, t_neh, t_palmer, t_campbell, t_random, t_spt, t_lpt))

        iteration += 1


if __name__ == '__main__':

    model = "ppo"

    model_path = "./result/model/ppo/0825_14_43_step150000_act.pt"
    data_path = "../environment/data/PBS_5_50.xlsx"

    log_dir = "./result/log"
    test_dir = "./result/test"

    if not os.path.exists(log_dir+ "/" + model):
        os.makedirs(log_dir + "/" + model)

    if not os.path.exists(test_dir+ "/" + model):
        os.makedirs(test_dir + "/" + model)

    params = {
        "model": model,
        "num_of_process": 5,
        "num_of_blocks": 50,
        "model_path": model_path,
        "log_dir": log_dir,
        "test_dir": test_dir,
        "n_embedding": 1024,
        "n_hidden": 512,
        "init_min": -0.08,
        "init_max": 0.08,
        "batch_size": 100000,
        "use_logit_clipping": False,
        "C": 10,
        "T": 1.5,
        "decode_type": "sampling",
        "n_glimpse": 1,
    }

    env = PanelBlockShop(params["num_of_process"], params["num_of_blocks"], distribution="uniform")
    # data = generate_block_data(num_of_process=params["num_of_process"], num_of_blocks=params["num_of_blocks"],
    #                            size=30, distribution="uniform")
    data = read_block_data(data_path)
    test_model(env, params, data)