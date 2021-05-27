import numpy as np
import pandas as pd


def Palmer(env, blocks):
    processing_time = blocks.cpu().numpy()
    block_num = processing_time.shape[0]
    process_num = processing_time.shape[1]

    index = np.zeros(block_num)
    for i, pt in enumerate(processing_time):
        for j in range(1, process_num + 1):
            index[i] += (2 * j - process_num - 1) * pt[j-1] / 2
    sequence = index.argsort()[::-1]
    makespan = env.show_result(blocks, sequence)
    return makespan


def Campbell(env, blocks):
    # processing_time = blocks.cpu().numpy()
    processing_time = blocks
    block_num = processing_time.shape[0]
    process_num = processing_time.shape[1]
    makespan_k = []
    seq_k = []

    for k in range(process_num - 1):
        seq = [0 for _ in range(block_num)]
        start = 0
        end = block_num - 1

        processing_time_johnson = pd.DataFrame(index=['blk_' + str(i) for i in range(block_num)], columns=['P1', 'P2'])
        for i in range(block_num):
            processing_time_johnson.iloc[i]['P1'] = sum([temp for temp in processing_time[i, :k + 1]])
            processing_time_johnson.iloc[i]['P2'] = sum([temp for temp in processing_time[i, k + 1:]])

        while len(processing_time_johnson):
            proc_time_min = np.min(processing_time_johnson)
            if proc_time_min['P1'] < proc_time_min['P2']:
                min_idx = np.argmin(processing_time_johnson['P1'])
                if type(min_idx) == list:
                    min_idx = min_idx[0]
                seq[start] = int(processing_time_johnson.index[min_idx][4:])
                processing_time_johnson.drop(processing_time_johnson.index[min_idx], inplace=True)
                start += 1
            elif proc_time_min['P1'] > proc_time_min['P2']:
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

        makespan_k.append(env.show_result(blocks, seq))
        seq_k.append(seq)

    min_makespan = np.min(makespan_k)
    return min_makespan



