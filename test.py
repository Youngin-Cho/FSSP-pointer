import torch
import random
import numpy as np

from time import time
from datetime import datetime
from env import PanelBlockShop
from config import Config, load_pkl, pkl_parser
from search import sampling, active_search


def search_tour(cfg, env, iter_num):
    date = datetime.now().strftime('%m%d_%H_%M')
    test_path = None
    for i in range(iter_num):
        blocks = env.get_blocks()

        # simplest way
        print('sampling ...')
        # t1 = time()
        sequence = sampling(cfg, env, blocks)
        # t2 = time()
        # print('%dmin %1.2fsec\n' % ((t2 - t1) // 60, (t2 - t1) % 60))
        C_RL = env.show_result(blocks, sequence)

        # random
        sequence = [i for i in range(cfg.block_num)]
        random.shuffle(sequence)
        sequence = np.array(sequence)
        C_RANDOM = env.show_result(blocks, sequence)

        # SPT
        processing_time = blocks.cpu().numpy().sum(axis=1)
        sequence = processing_time.argsort()
        C_SPT = env.show_result(blocks, sequence)

        # LPT
        processing_time = blocks.cpu().numpy().sum(axis=1)
        sequence = processing_time.argsort()[::-1]
        C_LPT = env.show_result(blocks, sequence)

        if test_path is None:
            test_path = cfg.test_dir + '%s_%s_results.csv' % (date, cfg.task)
            with open(test_path, 'w') as f:
                f.write('RL,RANDOM,SPT,LPT\n')
        else:
            with open(test_path, 'a') as f:
                f.write('%.1f,%.1f,%.1f,%.1f,\n' % (C_RL, C_RANDOM, C_SPT, C_LPT))

    # active search, update parameters during test
    # print('active search ...')
    # t1 = time()
    # pred_tour = active_search(cfg, env, test_input)
    # t2 = time()
    # print('%dmin %1.2fsec\n' % ((t2 - t1) // 60, (t2 - t1) % 60))
    # env.show(test_input, pred_tour)

    """
    # optimal solution, it takes time
    print('generate optimal solution ...')
    t1 = time()
    optimal_tour = env.get_optimal_tour(test_input)
    env.show(test_input, optimal_tour)
    t2 = time()
    print('%dmin %1.2fsec\n'%((t2-t1)//60, (t2-t1)%60))
    """


if __name__ == '__main__':
    cfg = load_pkl(pkl_parser().path)
    env = PanelBlockShop(cfg)

    # inputs = env.stack_nodes()
    # ~ tours = env.stack_random_tours()
    # ~ l = env.stack_l(inputs, tours)

    # ~ nodes = env.get_nodes(cfg.seed)
    # random_tour = env.get_random_tour()
    # ~ env.show(nodes, random_tour)

    # ~ env.show(inputs[0], random_tour)
    # ~ inputs = env.shuffle_index(inputs)
    # env.show(inputs[0], random_tour)

    # inputs = env.stack_nodes()
    # random_tour = env.get_random_tour()
    # env.show(inputs[0], random_tour)

    if cfg.mode == 'test':
        search_tour(cfg, env, 10)
    else:
        raise NotImplementedError('test only, specify test pkl file')