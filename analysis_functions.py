import numpy as np
import pickle
from scipy.linalg import norm
import pandas as pd

def signif(x, p):
    '''
    Function to round numpy array to p significant figures, from:
    https://stackoverflow.com/questions/18915378/rounding-to-significant-figures-in-numpy
    '''
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags
def get_arrays_for_gpe_verif(nroy_ex):
    """
    prepare nroy_sets rows for input to GPEs - just the four target params
    """
    arr = np.c_[nroy_ex[1:2].reshape(1,1),nroy_ex[3:].reshape(1,3)]
    return arr
def get_gpe_verif(no_sets, no_powers, nroy_sets, powervals, offset_name, gpe_run):
    """
    produce GPE predictions at NROY points
    """
    ver_results = np.zeros((no_sets, no_powers))
    for i in range(no_sets):
        for j in range(no_powers):
            nroy = get_arrays_for_gpe_verif(nroy_sets[i])
            gp = pickle.load(open("saved_gpes/fitted_GPE_ssfm"+
                                  str(int(powervals[j]))+offset_name+
                                  "_hipres"+gpe_run+".pkl", 'rb'))
            ver_results[i][j] = gp.predict(nroy)[0][0]  # select pred. mean
    return ver_results

def get_subset_inds(powernames):
    """
    obtain simulated data indices corresponding to a given launch power
    """
    subset = []
    for power in powernames:
        if power == '-112':
            subset.append(38)
        elif power == '-103':
            subset.append(47)
        elif power == '-94':
            subset.append(56)
        elif power == '-93':
            subset.append(57)
        elif power == '-85':
            subset.append(65)
        elif power == '-82':
            subset.append(68)
        elif power == '-74':
            subset.append(76)
        elif power == '-70':
            subset.append(80)
        elif power == '-68':
            subset.append(82)
        elif power == '-62':
            subset.append(88)
        elif power == '-61':
            subset.append(89)
        elif power == '-51':
            subset.append(99)
        elif power == '-45':
            subset.append(105)
        elif power == '-39':
            subset.append(111)
        elif power == '-34':
            subset.append(116)
        elif power == '-31':
            subset.append(119)
        elif power == '-27':
            subset.append(123)
        elif power == '-11':
            subset.append(139)
        elif power == '18':
            subset.append(168)
        elif power == '4':
            subset.append(154)
        elif power == '5':
            subset.append(155)
        elif power == '9':
            subset.append(159)
        elif power == '10':
            subset.append(160)
        elif power == '17':
            subset.append(167)
        elif power == '27':
            subset.append(177)
        elif power == '28':
            subset.append(178)
        elif power == '35':
            subset.append(185)
        elif power == '40':
            subset.append(190)
        elif power == '41':
            subset.append(191)
        elif power == '49':
            subset.append(199)
        elif power == '50':
            subset.append(200)
    return subset
def get_subset_inds_13(powernames):
    """
    obtain simulated data indices corresponding to a given launch power
    """
    subset = []
    for power in powernames:
        if power == '-112':
            subset.append(0)
        elif power == '-103':
            subset.append(1)
        elif power == '-93':
            subset.append(2)
        elif power == '-85':
            subset.append(3)
        elif power == '-68':
            subset.append(4)
        elif power == '-61':
            subset.append(5)
        elif power == '-11':
            subset.append(6)
        elif power == '27':
            subset.append(7)
        elif power == '28':
            subset.append(8)
        elif power == '40':
            subset.append(9)
        elif power == '41':
            subset.append(10)
        elif power == '49':
            subset.append(11)
        elif power == '50':
            subset.append(12)
    return subset
def intersection(datadirec, lop, pkp, hip, p4, p5, no_hm, run_name, offset_name, gpow_n, gpe_run):
    """
    obtain intersection of HM results at 3 launch powers, lo, pk, hi
    """
    if len(gpe_run) == 0:  # for consistency with previous file naming convention
        hm_lo = pickle.load(open(datadirec+"/hm_results"+lop+"_dbm_"
                                 +no_hm+"_3sig_noD"+gpow_n+run_name+offset_name+".pkl", 'rb'))
        hm_pk = pickle.load(open(datadirec+"/hm_results"+pkp+"_dbm_"
                                 +no_hm+"_3sig_noD"+gpow_n+run_name+offset_name+".pkl", 'rb'))
        hm_hi = pickle.load(open(datadirec+"/hm_results"+hip+"_dbm_"
                                 +no_hm+"_3sig_noD"+gpow_n+run_name+offset_name+".pkl", 'rb'))
        hm_4 = pickle.load(open(datadirec+"/hm_results"+p4+"_dbm_"
                                 +no_hm+"_3sig_noD"+gpow_n+run_name+offset_name+".pkl", 'rb'))
        hm_5 = pickle.load(open(datadirec+"/hm_results"+p5+"_dbm_"
                                 +no_hm+"_3sig_noD"+gpow_n+run_name+offset_name+".pkl", 'rb'))
    else:
        hm_lo = pickle.load(open(datadirec+"/hm_results"+lop+"_dbm_"
                +no_hm+"_3sig_noD"+gpow_n+run_name+offset_name+"_gpe"+gpe_run+".pkl", 'rb'))
        hm_pk = pickle.load(open(datadirec+"/hm_results"+pkp+"_dbm_"
                +no_hm+"_3sig_noD"+gpow_n+run_name+offset_name+"_gpe"+gpe_run+".pkl", 'rb'))
        hm_hi = pickle.load(open(datadirec+"/hm_results"+hip+"_dbm_"
                +no_hm+"_3sig_noD"+gpow_n+run_name+offset_name+"_gpe"+gpe_run+".pkl", 'rb'))
        hm_4 = pickle.load(open(datadirec+"/hm_results"+p4+"_dbm_"
                +no_hm+"_3sig_noD"+gpow_n+run_name+offset_name+"_gpe"+gpe_run+".pkl", 'rb'))
        hm_5 = pickle.load(open(datadirec+"/hm_results"+p5+"_dbm_"
                +no_hm+"_3sig_noD"+gpow_n+run_name+offset_name+"_gpe"+gpe_run+".pkl", 'rb'))
    for i in range(4):
        hm_lo[:,i] = signif(hm_lo[:,i],3)
        hm_pk[:,i] = signif(hm_pk[:,i],3)
        hm_hi[:,i] = signif(hm_hi[:,i],3)
        hm_4[:,i] = signif(hm_4[:,i],3)
        hm_5[:,i] = signif(hm_5[:,i],3)
    set_lo = set(map(tuple, hm_lo))
    set_pk = set(map(tuple, hm_pk))
    set_hi = set(map(tuple, hm_hi))
    set_4 = set(map(tuple, hm_4))
    set_5 = set(map(tuple, hm_5))
    intersec_arr = np.array(list(set_lo & set_pk & set_hi & set_4 & set_5))
    try:
        intersec_arr = np.c_[np.ones(len(intersec_arr)), intersec_arr[:,0], 17*np.ones(len(intersec_arr)), intersec_arr[:,1:]]
    except:
        print("intersec array length = "+str(len(intersec_arr)))
    print("intersec array length = "+str(len(intersec_arr)))
    return intersec_arr
def norms_test(gpow_n, powernames, intersec_gpe_preds):
    """
    find indices of best parameters from GPE predictions and return indices and norm values
    """
    if len(gpow_n) == 0:
        ground_truth_sim = pickle.load(open("gt_data/ssfm_full_power_ground_truth_sim_hipres.pkl", 'rb'))
        ground_truth_sim = ground_truth_sim.reshape(ground_truth_sim.shape[1],1)
    else:
        ground_truth_sim = pickle.load(open("gt_data/ssfm_full_power_ground_truth_sim_hipres"+gpow_n+".pkl", 'rb'))
        try:
            ground_truth_sim = ground_truth_sim.reshape(len(ground_truth_sim),1)
        except:
            ground_truth_sim = ground_truth_sim.reshape(ground_truth_sim.shape[1],1)
    if len(ground_truth_sim) == 211:
        subset_inds = get_subset_inds(powernames)
    else:
        subset_inds = get_subset_inds_13(powernames)
    snr_gt = ground_truth_sim[subset_inds].reshape(len(powernames),)
    l1_norm_ind, l1_norm_val  = get_l_norm(intersec_gpe_preds, snr_gt, 1)
    l2_norm_ind, l2_norm_val = get_l_norm(intersec_gpe_preds, snr_gt, 2)
    li_norm_ind, li_norm_val = get_l_norm(intersec_gpe_preds, snr_gt, np.inf)
    return l1_norm_ind, l1_norm_val, l2_norm_ind, l2_norm_val, li_norm_ind, li_norm_val

# def get_just_params(arr):

#     return np.c_[arr[1:2].reshape(1,1),arr[3:].reshape(1,3)].reshape(4,)
def get_append_arr(params1, params2, params3):
    """
    get the parameter results into correct format to append to the results
    """
    p1eqp2 = find_if_params_equal(params1, params2)
    p2eqp3 = find_if_params_equal(params2, params3)
    p1eqp3 = find_if_params_equal(params1, params3)
    if p1eqp2:
        if p1eqp3:  # 1 = 2 = 3
            return params1.reshape(1,len(params1)), np.array([123]).reshape(1,1)
        else:  # 1 = 2 != 3
            params = np.zeros([2,len(params1)])
            params[0] = params1
            params[1] = params3
            norms = np.zeros([2,1])
            norms[0] = 12
            norms[1] = 3
            return params, norms
    else: # 1 != 2
        if p1eqp3: # 1 = 3 != 2
            params = np.zeros([2,len(params1)])
            params[0] = params1
            params[1] = params2
            norms = np.zeros([2,1])
            norms[0] = 13
            norms[1] = 2
            return params, norms
        else:
            if p2eqp3: # 2 = 3 != 1
                params = np.zeros([2,len(params1)])
                params[0] = params1
                params[1] = params2
                norms = np.zeros([2,1])
                norms[0] = 1
                norms[1] = 23
                return params, norms
            else: # 1 != 2 != 3
                params = np.zeros([3,len(params1)])
                params[0] = params1
                params[1] = params2
                params[2] = params3
                norms = np.zeros([3,1])
                norms[0] = 1
                norms[1] = 2
                norms[2] = 3
                return params, norms
def generate_results(data_direc, lop, pkp, hip, p4, p5, no_hm, run_name,
                     offset_name, gpow_n, gpe_run, power_vals, powernames):
    """
    function to run results generation process
    """
    intersec_vals = intersection(data_direc, lop, pkp, hip, p4, p5, no_hm, run_name, offset_name,
                                gpow_n, gpe_run)
    num_sets = intersec_vals.shape[0]
    num_powers = len(power_vals)
    intersec_gpe_preds = get_gpe_verif(num_sets, num_powers, intersec_vals, power_vals, offset_name,
                                gpe_run)
    l1_ind, l1_norm, l2_ind, l2_norm, li_ind, li_norm = norms_test(gpow_n,
                                                powernames, intersec_gpe_preds)
    params_l1, params_l2, params_li = get_best_params(intersec_vals, l1_ind, l2_ind, li_ind)
    params, norms = get_append_arr(params_l1, params_l2, params_li)
    return params, norms, num_sets, l1_norm, l2_norm, li_norm, intersec_vals
def get_run_name(run):
    """
    deal with old naming convention
    """
    if run == '':
        return 1
    for i in range(2,6):
        if run == '_run' + str(i):
            return i
def append_results(data_direc, lop, pkp, hip, p4, p5, snr_pen, no_hm, offset_name, runs,
                   target_csv, gpe_run, gpow_n, power_vals, powernames):
    """
    controller function for generating results and appending them to csv
    """
    main_df = pd.read_csv(target_csv)
    for run in runs:
        params, norms, num_sets, l1_norm, l2_norm, li_norm, _ = generate_results(data_direc,
        lop, pkp, hip, p4, p5, no_hm, run, offset_name, gpow_n, gpe_run, power_vals, powernames)
        #print("L1, L2, LI ="+str(l1_norm)+","+str(l2_norm)+","+str(li_norm))
        if offset_name == '_asymr':
            off = 'R'
        elif offset_name == '_asyml':
            off = 'L'
        if gpe_run == '':
            gpe_name = 1
        else:
            gpe_name = gpe_run
        num_rows = params.shape[0] # number of rows to append to csv
        for j in range(num_rows):  # append each row at a time
            if len(gpow_n) == 0:
                data = [snr_pen, params[j][1], params[j][3], params[j][4], params[j][5], no_hm,
                        num_sets, int(norms[j][0]), off, get_run_name(run), gpe_name,
                        signif(l1_norm,4), signif(l2_norm,4), signif(li_norm,4) ]
                print(data)
                params_df = pd.DataFrame(data = np.array(data).reshape(1,14),
                                         columns=['SNRPENALTY','ALPHA','GAMMA','NF',
                         'BTB','NUMHM','NSETS','NORMS', 'ASYM','REPEATNUM','GPERNUM','L1', 'L2', 'LI'])
            else:
                data = [snr_pen, params[j][1], params[j][3], params[j][4], params[j][5], no_hm,
                    num_sets, int(norms[j][0]), off, get_run_name(run), gpe_name, gpow_n,
                        signif(l1_norm,4), signif(l2_norm,4), signif(li_norm,4)]
                print(data)
                params_df = pd.DataFrame(data = np.array(data).reshape(1,15),
                                     columns=['SNRPENALTY','ALPHA','GAMMA','NF',
                     'BTB','NUMHM','NSETS','NORMS', 'ASYM','REPEATNUM','GPERNUM','GPOW','L1', 'L2', 'LI'])
            main_df = main_df.append(params_df)
    main_df.to_csv(target_csv,index=False)

def drop_rows(target_csv, minind, maxind):
    """
    Function for programmtically dropping rows
    Note: indices in excel sheet are +2 compared to df
    """
    minind = minind - 2
    maxind = maxind - 2
    inds = [i for i in range(minind,maxind+1)][::-1]
    df = pd.read_csv(target_csv)
    for ind in inds:
        print(ind)
        df = df.drop(ind)
    df.to_csv(target_csv,index=False)
def get_param_for_norm(repseries, norm, param):
    """get index of desired norm from series,
    deal with stupid naming convention I chose
    """
    if norm == 1:
        if len(repseries[param][repseries['NORMS']==1]) != 0:
            return repseries[param][repseries['NORMS']==1]
        elif len(repseries[param][repseries['NORMS']==12])!= 0:
            return repseries[param][repseries['NORMS']==12]
        elif len(repseries[param][repseries['NORMS']==13])!= 0:
            return repseries[param][repseries['NORMS']==13]
        elif len(repseries[param][repseries['NORMS']==123])!= 0:
            return repseries[param][repseries['NORMS']==123]
    if norm == 2:
        if len(repseries[param][repseries['NORMS']==2]) != 0:
            return repseries[param][repseries['NORMS']==2]
        elif len(repseries[param][repseries['NORMS']==12])!= 0:
            return repseries[param][repseries['NORMS']==12]
        elif len(repseries[param][repseries['NORMS']==23])!= 0:
            return repseries[param][repseries['NORMS']==23]
        elif len(repseries[param][repseries['NORMS']==123])!= 0:
            return repseries[param][repseries['NORMS']==123]
    if norm == 3:
        if len(repseries[param][repseries['NORMS']==3]) != 0:
            return repseries[param][repseries['NORMS']==3]
        elif len(repseries[param][repseries['NORMS']==13])!= 0:
            return repseries[param][repseries['NORMS']==13]
        elif len(repseries[param][repseries['NORMS']==23])!= 0:
            return repseries[param][repseries['NORMS']==23]
        elif len(repseries[param][repseries['NORMS']==123])!= 0:
            return repseries[param][repseries['NORMS']==123]

def power_selector(snr_penalty, asym):
    if snr_penalty == 3001:
        if asym == '_asymr':
            powers = np.array([-103,-11,49,49,49],dtype=float) # 3 dB right 0.01 dB SNR pres
            snrpen = 3001
        else:
            powers = np.array([-112,-11,50, 50 ,50],dtype=float) # 3 dB left 0.01 dB SNR pres
            snrpen = 3001  # same as 0.1 dB pres
    elif snr_penalty == 2001:
        if asym == '_asymr':
            powers = np.array([-85,-11,40,40,40],dtype=float) # 2 dB right 0.01 dB SNR pres
            snrpen = 2001
        else:
            powers = np.array([-93,-11,41,41,41],dtype=float) # 2 dB left 0.01 dB SNR pres
            snrpen = 2001
    elif snr_penalty == 1001:
        if asym == '_asymr':
            powers = np.array([-61,-11,27,27,27],dtype=float) # 1 dB right 0.01 dB SNR pres
            snrpen = 1001
        else:
            powers = np.array([-68,-11,28,28,28],dtype=float) # 1 dB left 0.01 dB SNR pres
            snrpen = 1001
    elif snr_penalty == 0.5001:
        if asym == '_asymr':
            powers = np.array([-45,-11,18,18,18],dtype=float) # 1 dB right 0.01 dB SNR pres
            snrpen = 0.5001
        else:
            powers = np.array([-51,-11,17,17,17],dtype=float) # 1 dB left 0.01 dB SNR pres
            snrpen = 0.5001
    elif snr_penalty == 0.25001:
        if asym == '_asymr':
            powers = np.array([-34,-11,10,10,10],dtype=float) # 1 dB right 0.01 dB SNR pres
            snrpen = 0.25001
        else:
            powers = np.array([-39,-11,9,9,9],dtype=float) # 1 dB left 0.01 dB SNR pres
            snrpen = 0.25001
    return powers, snrpen

def get_best_params(intersec_vals, l1_norm_ind, l2_norm_ind, li_norm_ind):
    """
    obtain parameters from best indices
    """
    return intersec_vals[l1_norm_ind], intersec_vals[l2_norm_ind], intersec_vals[li_norm_ind]
def get_l_norm(intersec_gpe_preds, snr_in, nord):
    """
    calculate l-x norm minimum of gpe predictions with respect to simulated data targets
    """
    l_norms = []
    for i in range(len(intersec_gpe_preds)):
        lnorm = norm(intersec_gpe_preds[i] - snr_in, ord=nord)
        l_norms.append(lnorm)
    #print("min ind = " + str(np.argmin(l_norms)) + ", l"+ str(nord) +" norm = " + str(np.min(l_norms)))
    return np.argmin(l_norms), np.min(l_norms)
def find_if_params_equal(params1, params2):
    """
    determine whether or not parameters are equal, to help maintain data structure convention
    """
    if len(np.where(params1 != params2)[0]) == 0:  # if one of the params is different
        return True
    else:
        return False

    
    

