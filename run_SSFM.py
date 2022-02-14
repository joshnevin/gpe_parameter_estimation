import numpy as np
import mogp_emulator
import pickle
import matlab.engine
from numpy.random import normal
import os
import argparse

def get_arrays_for_full_ssfm(hm_arr, powers):
    num_sets = hm_arr.shape[0]
    arr = hm_arr.reshape(1,num_sets)*np.ones([211,num_sets])
    arr[:,0] = powers
    return arr

parser = argparse.ArgumentParser(description='Set up path')
parser.add_argument('--path', default='n15_n1_5_dbm_2M_3sig_newrange_run2', type=str)
parser.add_argument('--sam_num', default='', type=str)
args = parser.parse_args()

file_path = args.path
sam_num = args.sam_num

nroy_sets = pickle.load(open("intersec_res/intersec_results_"+file_path+".pkl", 'rb'))
power_exp = np.arange(-15., 6.1, 0.1)
num_powers = power_exp.shape[0]
num_sets = nroy_sets.shape[0]

ssfm_results = np.zeros((num_sets, num_powers))

print("Starting MATLAB engine...")
eng = matlab.engine.start_matlab()

cwd = os.getcwd()
path = cwd + '/SSF'
eng.addpath(path,nargout=0)

for i in range(num_sets):
    nroy = get_arrays_for_full_ssfm(nroy_sets[i], power_exp)
    for j in range(num_powers):
        ssfm_results[i][j] = eng.SSFM_f(nroy[j][0].item(), nroy[j][1].item(), nroy[j][2].item(), nroy[j][3].item(),
        nroy[j][4].item(), nroy[j][5].item(), nargout=1)

eng.quit()
print("Quit MATLAB engine.")

pickle.dump(ssfm_results, open("gt_data/ssfm_full_power_"+file_path+"_hipres"+sam_num+".pkl", 'wb')) # save simulation outputs
