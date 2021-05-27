import numpy as np


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
    processing_time = blocks.cpu().numpy()
    block_num = processing_time.shape[0]
    process_num = processing_time.shape[1]

    for k in range(process_num - 1):
        seq = np.zeros(block_num)
        start = 0
        end = block_num - 1
        processing_time_johnson = np.zeros((block_num, 2))
        for i in range(block_num):
            processing_time_johnson[i, 0] = sum([temp for temp in processing_time[i, :k + 1]])
            processing_time_johnson[i, 1] = sum([temp for temp in processing_time[i, k + 1:]])

