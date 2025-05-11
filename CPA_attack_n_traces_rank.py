import numpy as np
import struct
import time
import os
from decimal import Decimal


def get_weights_arr(total_n_weights):
    weights_arr = []
    w_val = Decimal('-2.0')
    step = Decimal('0.01')
    for i in range (total_num_of_weight):
        weights_arr.append(float(w_val))
        w_val += step
    return weights_arr

total_num_of_weight = 401
weights_arr = get_weights_arr(total_num_of_weight)
#the weight value to be recovered
true_weight = 1.43
true_weight_index = weights_arr.index(true_weight)
print(f'index of the true weight value: {true_weight_index}')

def float_to_binary_str(f):
    # Pack the float into 4 bytes (32-bit) using IEEE 754 standard
    [packed] = struct.unpack('!I', struct.pack('!f', f))
    # Convert the packed number to a binary string
    return f"{packed:032b}"

def HW_float32(f):
    # Get the binary representation of the 32-bit float
    binary_str = float_to_binary_str(f)
    # Count and return the number of '1' bits
    return binary_str.count('1')

#get one part of the binary representation
def getbyte(f,byte_position):
    if byte_position == 0:#sign bit
        inbinary = float_to_binary_str(f)[0]
    elif byte_position == 4:#last 7 bits
        inbinary = float_to_binary_str(f)[25:32]
    else:
        inbinary = float_to_binary_str(f)[(byte_position-1)*8+1:byte_position*8+1]
    
    return int(inbinary,2)

#true values of different parts of the weight
true_sign_bit = getbyte(true_weight,0)
true_exponent_bits = getbyte(true_weight,1)
true_byte_two = getbyte(true_weight,2)
true_byte_three = getbyte(true_weight,3)
true_byte_four = getbyte(true_weight,4)
true_value_index = weights_arr.index(true_weight)

print("sign bit = " + str(true_sign_bit) + ", exponent = " + str(true_exponent_bits) + ", first mantissa byte = " + str(true_byte_two) + ", second mantissa byte = " + str(true_byte_three) + ", last 7 bits = " + str(true_byte_four))

def get_hypothetical_leakages(num_of_traces, inputs_arr):
    time_start = time.time()
    print(f'Calculating Hypothetical Leakages... - start: {time_start}')
    hypothetical_leakages = [[] for i in range(total_num_of_weight)]
    for weight_index in range(total_num_of_weight):
        for j in range(num_of_traces):
            #compute the hypothetical product values for each hypothetical value of weight and each random input
            hypothetical_product = weights_arr[weight_index] * inputs_arr[j]
            #compute hypothetical leakages
            hypothetical_leakages[weight_index].append(HW_float32(hypothetical_product))
    print(f'Finished Calculating Hypothetical Leakages in {time.time() - time_start}')
    return np.array(hypothetical_leakages)

def CPA_attack(num_of_traces, inputs_arr, total_time_sample, start_time_sample, trace_arr):
    r_abs = [[] for i in range(total_num_of_weight)]
    #get hypothetical leakages
    hypothetical_leakages = get_hypothetical_leakages(num_of_traces, inputs_arr)
    for t in range(total_time_sample):
        time_sample_index = t + start_time_sample
        if time_sample_index % 1000 == 0:
            print(f'num_of_traces={num_of_traces} ({start_time_sample}-{start_time_sample + total_time_sample}) time samples - calculating for time sample {str(time_sample_index)}')
        for weight_index in range(total_num_of_weight):
            corr_coef = np.corrcoef(hypothetical_leakages[weight_index][:num_of_traces],trace_arr[:num_of_traces,time_sample_index])[0][1]
            r_abs[weight_index].append(abs(corr_coef))
    return r_abs

def recover_byte(correlation_arr, byte_position, total_time_sample):
    num_of_different_values = 256 #for byte two and three and exponent bits, we have 256 different values
    if byte_position == 0:#for sign bit, there are just 2 different values
        num_of_different_values = 2
    if byte_position == 4:#for last 7 bits, there are 2^7 different values
        num_of_different_values = 128
    correlations = [[0 for t in range(total_time_sample)] for i in range(num_of_different_values)]
    for weight_index in range(total_num_of_weight):#for each weight
        byte_value = getbyte(weights_arr[weight_index],byte_position)#for the chosen byte of this weight value
        for t in range(total_time_sample):#take the correlations for this weight value, for each time sample, update the correlations
            if correlation_arr[weight_index][t] > correlations[byte_value][t]:
                correlations[byte_value][t] = correlation_arr[weight_index][t]
        
    return correlations

def load_traces(num_of_traces, folder_name, max_time_sample):
    trace_waves_arr = []
    inputs_arr = []

    for i in range(num_of_traces):
        with open(folder_name + '/trace_'+str(i)+'.txt') as f:
            lines = f.read().splitlines()
            trace_waves_arr.append(lines[:max_time_sample])
        if (i+1) % 1000 == 0:
            print(f'loaded {(i+1)} traces')

    with open(folder_name + '/inputs.txt') as f:
        inputs_arr = f.read().splitlines()

    time_start = time.time()
    print(f'Converting trace_waves_arr to np.array - start: {time_start}')
    trace_waves_arr = np.array(trace_waves_arr, dtype=float)
    print(f'Finished converting trace_waves_arr to np.array in: {time.time() - time_start}')
    #trace_waves_arr = np.array(trace_waves_arr)
    #trace_waves_arr = trace_waves_arr.astype(float)
    time_start = time.time()
    print(f'Converting inputs_arr to np.array - start: {time_start}')
    inputs_arr = np.array(inputs_arr[:num_of_traces], dtype=float)
    print(f'Finished converting inputs_arr to np.array in: {time.time() - time_start}')
    #inputs_arr = np.array(inputs_arr[:num_of_traces])
    #inputs_arr = inputs_arr.astype(float)

    return trace_waves_arr, inputs_arr

def load_traces2(num_of_traces, folder_name, max_time_sample):
    trace_waves_arr = np.zeros((num_of_traces, max_time_sample), dtype=float)
    inputs_arr = []

    for i in range(num_of_traces):
        with open(folder_name + '/trace_'+str(i)+'.txt') as f:
            lines = f.read().splitlines()
            trace_waves_arr[i] = np.array(lines[:max_time_sample], dtype=float)
        if (i+1) % 1000 == 0:
            print(f'loaded {(i+1)} traces')
    print(f'Traces Loaded!')
    print(f'Loading Inputs...')
    with open(folder_name + '/inputs.txt') as f:
        inputs_arr = f.read().splitlines()
    inputs_arr = np.array(inputs_arr[:num_of_traces], dtype=float)
    print(f'Inputs Loaded!')
    return trace_waves_arr, inputs_arr


def find_max_and_max_true(correlation_arr, true_value_index):
    overall_max = 0
    true_max = 0
    total_plots = len(correlation_arr)
    
    for i in range(total_plots):
        if i == true_value_index:
            true_max = max(correlation_arr[i])
        else:
            w_max = max(correlation_arr[i])
            if w_max > overall_max:
                overall_max = w_max
                
    return overall_max, true_max


def get_true_rank(correlation_arr, true_value_index):

    length = len(correlation_arr)
    max_correlations = np.zeros((length,), dtype=float)
    for i in range(length):
        max_correlations[i] = max(correlation_arr[i])

    sorted_indices = (-max_correlations).argsort()
    index = np.where(sorted_indices == true_value_index)[0][0]
    return index


def run_CPA_n_traces(inputs_arr, total_ts, start_ts, trace_arr, nlow, nhigh, nincrement, save_folder, backup_interval):
    # initialize np arrays for true weight rankings
    arr_num_traces = np.array([])
    arr_true_rank_overall = np.array([])
    arr_true_rank_sign_bit = np.array([])
    arr_true_rank_exponent_bits = np.array([])
    arr_true_rank_byte_two = np.array([])
    arr_true_rank_byte_three = np.array([])
    arr_true_rank_byte_four = np.array([])

    num_of_traces_low = nlow
    num_of_traces_high = nhigh
    increment = nincrement

    completed = 0
    start_time = time.time()
    for i in range(num_of_traces_low, (num_of_traces_high + 1), increment):
        print(f'Running CPA for num_of_traces={i}')
        r = CPA_attack(num_of_traces=i, inputs_arr=inputs_arr, total_time_sample=total_ts, start_time_sample=start_ts, trace_arr=trace_arr)

        print(f'CPA attack for num_of_traces={i} and ({start}-{start + total_ts}) time samples complete in {time.time() - start_time}!')
        correlations_sign_bit = recover_byte(r, 0, total_ts)
        correlations_exponent_bits = recover_byte(r,1, total_ts)
        correlations_byte_two = recover_byte(r, 2, total_ts)
        correlations_byte_three = recover_byte(r, 3, total_ts)
        correlations_byte_four = recover_byte(r, 4, total_ts)

        true_rank_overall = get_true_rank(r, true_weight_index)
        true_rank_sign_bit = get_true_rank(correlations_sign_bit, true_sign_bit)
        true_rank_exponent_bits = get_true_rank(correlations_exponent_bits, true_exponent_bits)
        true_rank_byte_two = get_true_rank(correlations_byte_two, true_byte_two)
        true_rank_byte_three = get_true_rank(correlations_byte_three, true_byte_three)
        true_rank_byte_four = get_true_rank(correlations_byte_four, true_byte_four)

        
        arr_num_traces = np.append(arr_num_traces, i)
        arr_true_rank_overall = np.append(arr_true_rank_overall, true_rank_overall)
        arr_true_rank_sign_bit = np.append(arr_true_rank_sign_bit, true_rank_sign_bit)
        arr_true_rank_exponent_bits = np.append(arr_true_rank_exponent_bits, true_rank_exponent_bits)
        arr_true_rank_byte_two = np.append(arr_true_rank_byte_two, true_rank_byte_two)
        arr_true_rank_byte_three = np.append(arr_true_rank_byte_three, true_rank_byte_three)
        arr_true_rank_byte_four = np.append(arr_true_rank_byte_four, true_rank_byte_four)

        print(arr_num_traces)
        print(f'Overall: {true_rank_overall}\nSign Bit: {true_rank_sign_bit}\nExponent Bits: {true_rank_exponent_bits}\nByte Two: {true_rank_byte_two}\nByte Three: {true_rank_byte_three}\nByte Four: {true_rank_byte_four}')

        completed += 1
        if completed % backup_interval == 0:
            np.save(save_folder + f'arr_num_traces_{nlow}_{i}_{nincrement}.npy', arr_num_traces)
            np.save(save_folder + f'arr_true_rank_overall_{nlow}_{i}_{nincrement}.npy', arr_true_rank_overall)
            np.save(save_folder + f'arr_true_rank_sign_bit_{nlow}_{i}_{nincrement}.npy', arr_true_rank_sign_bit)
            np.save(save_folder + f'arr_true_rank_exponent_bits_{nlow}_{i}_{nincrement}.npy', arr_true_rank_exponent_bits)
            np.save(save_folder + f'arr_true_rank_byte_two_{nlow}_{i}_{nincrement}.npy', arr_true_rank_byte_two)
            np.save(save_folder + f'arr_true_rank_byte_three_{nlow}_{i}_{nincrement}.npy', arr_true_rank_byte_three)
            np.save(save_folder + f'arr_true_rank_byte_four_{nlow}_{i}_{nincrement}.npy', arr_true_rank_byte_four)
            np.save(save_folder + f'arr_num_traces_{nlow}_{i}_{nincrement}.npy', arr_num_traces)

    np.save(save_folder + f'final_arr_num_traces_{nlow}_{i}_{nincrement}.npy', arr_num_traces)
    np.save(save_folder + f'final_arr_true_rank_overall_{nlow}_{i}_{nincrement}.npy', arr_true_rank_overall)
    np.save(save_folder + f'final_arr_true_rank_sign_bit_{nlow}_{i}_{nincrement}.npy', arr_true_rank_sign_bit)
    np.save(save_folder + f'final_arr_true_rank_exponent_bits_{nlow}_{i}_{nincrement}.npy', arr_true_rank_exponent_bits)
    np.save(save_folder + f'final_arr_true_rank_byte_two_{nlow}_{i}_{nincrement}.npy', arr_true_rank_byte_two)
    np.save(save_folder + f'final_arr_true_rank_byte_three_{nlow}_{i}_{nincrement}.npy', arr_true_rank_byte_three)
    np.save(save_folder + f'final_arr_true_rank_byte_four_{nlow}_{i}_{nincrement}.npy', arr_true_rank_byte_four)




start = 490
end_pro = 4301
total_pro = end_pro - start

remote_dir = "/home/leonard/txt_traces/protected100k-2"
local_dir =  "./txt_traces/project-protected-10k-decimate1_new_weights" #"./txt_traces/project-unprotected-10k-decimate1"


traces_pro, inputs_pro = load_traces2(num_of_traces=50000, folder_name = remote_dir, max_time_sample=end_pro)
#r_pro = CPA_attack(num_of_traces=10000, inputs_arr=inputs_pro, total_time_sample=total_pro, start_time_sample=start, trace_arr=traces_pro)

run_CPA_n_traces(inputs_arr = inputs_pro,
                total_ts = total_pro,
                start_ts = start,
                trace_arr = traces_pro,
                nlow = 10,
                nhigh = 50000,
                nincrement = 10,
                save_folder = "./attack_out_rank_pro/",
                backup_interval = 100)

