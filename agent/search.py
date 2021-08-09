import os
import torch
import numpy as np
import pandas as pd

from actor import PtrNet1


def sampling(env, params, test_input):
    test_inputs = test_input.repeat(params["batch_size"], 1, 1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = PtrNet1(params)
    if os.path.exists(params["model_path"]):
        model.load_state_dict(torch.load(params["model_path"], map_location=device))
    else:
        print('specify pretrained model path')
    model = model.to(device)

    pred_sequence, _ = model(test_inputs, device)
    makespan_batch = env.stack_makespan(test_inputs, pred_sequence)
    index_makespan_min = torch.argmin(makespan_batch)
    best_sequence = pred_sequence[index_makespan_min]
    best_makespan = makespan_batch[index_makespan_min]
    return best_sequence, best_makespan


def Palmer_sequence(env, test_input):
    test_input = test_input.cpu().numpy()

    index = np.zeros(env.num_of_blocks)
    for i, processing_time in enumerate(test_input):
        for j in range(1, env.num_of_process + 1):
            index[i] += (2 * j - env.num_of_process - 1) * processing_time[j-1] / 2

    sequence = index.argsort()[::-1]
    makespan = env.calculate_makespan(test_input, sequence).item()
    return sequence, makespan


def Campbell_sequence(env, test_input):
    processing_time = test_input.cpu().numpy()
    makespan_k = []
    sequence_k = []

    for k in range(env.num_of_process - 1):
        seq = [0 for _ in range(env.num_of_blocks)]
        start = 0
        end = env.num_of_blocks - 1

        processing_time_johnson = pd.DataFrame(index=['blk_' + str(i) for i in range(env.num_of_blocks)],
                                               columns=['P1', 'P2'])
        for i in range(env.num_of_blocks):
            processing_time_johnson.iloc[i]['P1'] = sum([temp for temp in processing_time[i, :k + 1]])
            processing_time_johnson.iloc[i]['P2'] = sum([temp for temp in processing_time[i, k + 1:]])

        while len(processing_time_johnson):
            processing_time_min = np.min(processing_time_johnson)
            if processing_time_min['P1'] <= processing_time_min['P2']:
                min_idx = np.argmin(processing_time_johnson['P1'])
                if type(min_idx) == list:
                    min_idx = min_idx[0]
                seq[start] = int(processing_time_johnson.index[min_idx][4:])
                processing_time_johnson.drop(processing_time_johnson.index[min_idx], inplace=True)
                start += 1
            elif processing_time_min['P1'] > processing_time_min['P2']:
                min_idx = np.argmin(processing_time_johnson['P2'])
                if type(min_idx) == list:
                    min_idx = min_idx[0]
                seq[end] = int(processing_time_johnson.index[min_idx][4:])
                processing_time_johnson.drop(processing_time_johnson.index[min_idx], inplace=True)
                end -= 1
            else:
                min_P1_idx = np.argmin(processing_time_johnson['P1'])
                if type(min_P1_idx) == list:
                    min_P1_idx = min_P1_idx[0]
                seq[start] = int(processing_time_johnson.index[min_P1_idx][4:])
                processing_time_johnson.drop(processing_time_johnson.index[min_P1_idx], inplace=True)
                start += 1

                if len(processing_time_johnson):
                    min_P2_idx = np.argmin(processing_time_johnson['P2'])
                    if type(min_P2_idx) == list:
                        min_P2_idx = min_P2_idx[0]
                    seq[end] = int(processing_time_johnson.index[min_P2_idx][4:])
                    processing_time_johnson.drop(processing_time_johnson.index[min_P2_idx], inplace=True)
                    end -= 1

        makespan_k.append(env.calculate_makespan(test_input, seq).item())
        sequence_k.append(seq)

    best_sequence = sequence_k[np.argmin(makespan_k)]
    best_makespan = np.min(makespan_k)

    return best_sequence, best_makespan


def random_sequence(env, test_input):
    sequence = np.random.permutation(env.num_of_blocks)
    makespan = env.calculate_makespan(test_input, sequence).item()
    return sequence, makespan


def LPT_sequence(env, test_input):
    processing_time = test_input.cpu().numpy().sum(axis=1)
    sequence = processing_time.argsort()
    makespan = env.calculate_makespan(processing_time, sequence).item()
    return sequence, makespan


def SPT_sequence(env, test_input):
    processing_time = test_input.cpu().numpy().sum(axis=1)
    sequence = processing_time.argsort()[::-1]
    makespan = env.calculate_makespan(processing_time, sequence).item()
    return sequence, makespan