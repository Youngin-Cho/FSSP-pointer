import torch
import simpy
import math
import itertools

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from SimComponents import *


class PanelBlockShop():
    def __init__(self, cfg):
        '''
        nodes(cities) : contains nodes and their 2 dimensional coordinates
        [city_t, 2] = [3,2] dimension array e.g. [[0.5,0.7],[0.2,0.3],[0.4,0.1]]
        '''
        self.batch = cfg.batch
        self.block_num = cfg.block_num
        self.process_num = 6

    def get_blocks(self, seed=None):
        '''
        return panel_block:(block_num,process_num=6)
        '''
        if seed is not None:
            torch.manual_seed(seed)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        shape = [0.543, 0.525, 0.196, 0.451, 0.581, 0.432]
        scale = [2.18, 2.18, 0.518, 2.06, 1.79, 2.10]

        process_time = np.zeros((self.block_num, self.process_num))
        for i in range(self.process_num):
            r = np.round(stats.lognorm.rvs(shape=shape[i], loc=0, scale=scale[i], size=self.block_num), 1)
            process_time[:, i] = r

        return torch.FloatTensor(process_time, device=device)

    def stack_blocks(self):
        '''
        nodes:(block_num,process_num)
        return inputs:(batch,block_num,process_num)
        '''
        list = [self.get_blocks() for i in range(self.batch)]
        inputs = torch.stack(list, dim=0)
        return inputs

    def get_batch_blocks(self, n_samples, seed=None):
        '''
        return nodes:(batch,block_num,process_num)
        '''
        if seed is not None:
            torch.manual_seed(seed)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        shape = [0.543, 0.525, 0.196, 0.451, 0.581, 0.432]
        scale = [2.18, 2.18, 0.518, 2.06, 1.79, 2.10]

        process_time = np.zeros((n_samples * self.block_num, self.process_num))
        for i in range(self.process_num):
            r = np.round(stats.lognorm.rvs(shape[i], loc=0, scale=scale[i], size=n_samples * self.block_num), 1)
            process_time[:, i] = r
        process_time = process_time.reshape((n_samples, self.block_num, self.process_num))

        return torch.FloatTensor(process_time, device=device)

    def stack_random_sequence(self):
        '''
        sequence:(block_num)
        return tours:(batch,block_num)
        '''
        list = [self.get_random_sequence() for i in range(self.batch)]
        sequences = torch.stack(list, dim=0)
        return sequences

    def stack_C(self, inputs, sequences):
        '''
        inputs:(batch,city_t,2)
        tours:(batch,city_t)
        return l_batch:(batch)
        '''
        list = [self.get_makespan(inputs[i], sequences[i]) for i in range(self.batch)]
        C_batch = torch.stack(list, dim=0)
        return C_batch

    def show(self, nodes, tour):
        nodes = nodes.cpu().detach()
        print('distance:{:.3f}'.format(self.get_tour_distance(nodes, tour)))
        print(tour)
        plt.figure()
        plt.plot(nodes[:, 0], nodes[:, 1], 'yo', markersize=16)
        np_tour = tour[:].cpu().detach()
        np_fin_tour = [tour[-1].item(), tour[0].item()]
        plt.plot(nodes[np_tour, 0], nodes[np_tour, 1], 'k-', linewidth=0.7)
        plt.plot(nodes[np_fin_tour, 0], nodes[np_fin_tour, 1], 'k-', linewidth=0.7)
        for i in range(self.city_t):
            plt.text(nodes[i, 0], nodes[i, 1], str(i), size=10, color='b')
        plt.show()

    def shuffle(self, inputs):
        '''
        shuffle nodes order with a set of xy coordinate
        inputs:(batch,block_num,process_num)
        return shuffle_inputs:(batch,block_num,process_num)
        '''
        shuffle_inputs = torch.zeros(inputs.size())
        for i in range(self.batch):
            perm = torch.randperm(self.block_num)
            shuffle_inputs[i, :, :] = inputs[i, perm, :]
        return shuffle_inputs

    def back_tours(self, pred_shuffle_tours, shuffle_inputs, test_inputs, device):
        '''
        pred_shuffle_tours:(batch,city_t)
        shuffle_inputs:(batch,city_t_t,2)
        test_inputs:(batch,city_t,2)
        return pred_tours:(batch,city_t)
        '''
        pred_tours = []
        for i in range(self.batch):
            pred_tour = []
            for j in range(self.city_t):
                xy_temp = shuffle_inputs[i, pred_shuffle_tours[i, j]].to(device)
                for k in range(self.city_t):
                    if torch.all(torch.eq(xy_temp, test_inputs[i, k])):
                        pred_tour.append(torch.tensor(k))
                        if len(pred_tour) == self.city_t:
                            pred_tours.append(torch.stack(pred_tour, dim=0))
                        break
        pred_tours = torch.stack(pred_tours, dim=0)
        return pred_tours

    def get_makespan(self, blocks, sequence):
        '''
        blocks:(block_num,process_num), sequence:(block_num)
        C(= total makespan)
        return C:(1)
        '''

        columns = pd.MultiIndex.from_product([[i for i in range(self.process_num + 1)],
                                              ['start_time', 'process_time', 'process']])
        df = pd.DataFrame(columns=columns, index=[i for i in range(self.block_num)])
        for i in range(self.process_num + 1):
            if i == self.process_num:
                df[(i, 'start_time')] = None
                df[(i, 'process_time')] = None
                df[(i, 'process')] = 'Sink'
            else:
                df[(i, 'start_time')] = 0.0
                df[(i, 'process_time')] = blocks[:, i].cpu().numpy()
                df[(i, 'process')] = "Process{0}".format(i)
        panel_blocks = [Part(i, df.loc[i]) for i in sequence.cpu().numpy()]

        env = simpy.Environment()
        model = {}

        event_path = "./simulation"
        if not os.path.exists(event_path):
            os.makedirs(event_path)

        monitor = Monitor(event_path + '/event_PBS.csv')
        source = Source(env, "Source", panel_blocks, model, monitor)
        for i in range(self.process_num + 1):
            model['Process{0}'.format(i)] = Process(env, 'Process{0}'.format(i), 1, model, monitor, qlimit=1)
            if i == self.process_num:
                model['Sink'] = Sink(env, 'Sink', monitor)

        env.run()
        C = model["Sink"].last_arrival

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        return torch.FloatTensor([C], device=device)

    def get_random_sequence(self):
        '''
        return tour:(block_num)
        '''
        sequence = []
        while set(sequence) != set(range(self.block_num)):
            block = np.random.randint(self.block_num)
            if block not in sequence:
                sequence.append(block)
        sequence = torch.from_numpy(np.array(sequence))
        return sequence