import math
import random


def long_range_rand(seq_len, num_seg):
    r = int(seq_len / num_seg)
    real_num_seg = int(math.ceil(seq_len / r))

    frame_ind = []
    for i in range(0, real_num_seg-1):
        frame_ind.append(random.randint(i*r, (i+1)*r-1))
    frame_ind.append(random.randint((real_num_seg-1)*r, seq_len-1))

    frame_ind = frame_ind[len(frame_ind)-num_seg:]
    return frame_ind

def long_range_first(seq_len, num_seg):
    r = int(seq_len / num_seg)

    frame_ind = []
    for i in range(0, seq_len, r):
        frame_ind.append(i)
    frame_ind = frame_ind[len(frame_ind)-num_seg:]
    return frame_ind

def long_range_last(seq_len, num_seg):
    r = int(seq_len / num_seg)

    frame_ind = []
    for i in range(seq_len-1, -1, -r):
        frame_ind.append(i)

    frame_ind = frame_ind[0:num_seg]
    frame_ind.reverse()
    return frame_ind

def long_range_sample(seq_len, num_seg, mode):
    if mode == "random":
        return long_range_rand(seq_len, num_seg)
    elif mode == "first":
        return long_range_first(seq_len, num_seg)
    elif mode == "last":
        return long_range_last(seq_len, num_seg)
    else:
        raise Exception(f"Given mode is unacceptable!")
