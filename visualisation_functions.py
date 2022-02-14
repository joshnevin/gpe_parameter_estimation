import numpy as np
import pickle
from scipy.linalg import norm
import pandas as pd
from analysis_functions import (signif, get_arrays_for_gpe_verif, get_gpe_verif, get_subset_inds,intersection,
norms_test, get_best_params, get_append_arr, generate_results, get_run_name, append_results, power_selector,find_if_params_equal)

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
def get_l_norm_all_values(intersec_gpe_preds, snr_in, nord):
    """
    calculate l-x norm of gpe predictions with respect to simulated data targets
    """
    l_norms = []
    for i in range(len(intersec_gpe_preds)):
        lnorm = norm(intersec_gpe_preds[i] - snr_in, ord=nord)
        l_norms.append(lnorm)
    #print("min ind = " + str(np.argmin(l_norms)) + ", l"+ str(nord) +" norm = " + str(np.min(l_norms)))
    return l_norms

def find_if_params_equal(params1, params2):
    """
    determine whether or not parameters are equal, to help maintain data structure convention
    """
    if len(np.where(params1 != params2)[0]) == 0:  # if one of the params is different
        return True
    else:
        return False

def get_norm_val_for_norm(repseries, norm):
    """get index of desired norm from series,
    deal with stupid naming convention I chose
    """
    if norm == 1:
        if len(repseries['L1'][repseries['NORMS']==1]) != 0:
            return repseries['L1'][repseries['NORMS']==1]
        elif len(repseries['L1'][repseries['NORMS']==12])!= 0:
            return repseries['L1'][repseries['NORMS']==12]
        elif len(repseries['L1'][repseries['NORMS']==13])!= 0:
            return repseries['L1'][repseries['NORMS']==13]
        elif len(repseries['L1'][repseries['NORMS']==123])!= 0:
            return repseries['L1'][repseries['NORMS']==123]
    if norm == 2:
        if len(repseries['L2'][repseries['NORMS']==2]) != 0:
            return repseries['L2'][repseries['NORMS']==2]
        elif len(repseries['L2'][repseries['NORMS']==12])!= 0:
            return repseries['L2'][repseries['NORMS']==12]
        elif len(repseries['L2'][repseries['NORMS']==23])!= 0:
            return repseries['L2'][repseries['NORMS']==23]
        elif len(repseries['L2'][repseries['NORMS']==123])!= 0:
            return repseries['L2'][repseries['NORMS']==123]
    if norm == 3:
        if len(repseries['L3'][repseries['NORMS']==3]) != 0:
            return repseries['L3'][repseries['NORMS']==3]
        elif len(repseries['L3'][repseries['NORMS']==13])!= 0:
            return repseries['L3'][repseries['NORMS']==13]
        elif len(repseries['L3'][repseries['NORMS']==23])!= 0:
            return repseries['L3'][repseries['NORMS']==23]
        elif len(repseries['L3'][repseries['NORMS']==123])!= 0:
            return repseries['L3'][repseries['NORMS']==123]
def get_error(df, snr_pen, param, asym, nohm, numsam, norm, samples=True):
    if numsam == 1:  # required to deal with stupid naming convention...
        if samples:
            reps = df.loc[df['NUMHM'] == nohm].loc[df['SNRPENALTY'] == snr_pen].loc[
            df['ASYM'] == asym].loc[
            df['GPOW'] == '_gpow_ave'+str(numsam)]
        else:
            reps = df.loc[df['NUMHM'] == nohm].loc[df['SNRPENALTY'] == snr_pen].loc[
            df['ASYM'] == asym].loc[
            df['GPOW'] == '_gpow_var'+str(numsam)]
    else:
        if samples:
            reps = df.loc[df['NUMHM'] == nohm].loc[df['SNRPENALTY'] == snr_pen].loc[
            df['ASYM'] == asym].loc[
            df['GPOW'] == '_gpow_av'+str(numsam)]
        else:
            reps = df.loc[df['NUMHM'] == nohm].loc[df['SNRPENALTY'] == snr_pen].loc[
            df['ASYM'] == asym].loc[
            df['GPOW'] == '_gpow_var'+str(numsam)]
    if param == 'ALPHA':
        return (get_param_for_norm(reps, norm, param) - 0.2).to_numpy()
    elif param == 'GAMMA':
        return (get_param_for_norm(reps, norm, param) - 1.2).to_numpy()
    elif param == 'NF':
        return (get_param_for_norm(reps, norm, param) - 4.5).to_numpy()
    elif param == 'BTB':
        return (get_param_for_norm(reps, norm, param) - 14.8).to_numpy()

def get_norm_error(df, snr_pen, param, asym, nohm, numsam, norm, samples=True):
    if numsam == 1:  # required to deal with stupid naming convention...
        if samples:
            reps = df.loc[df['NUMHM'] == nohm].loc[df['SNRPENALTY'] == snr_pen].loc[
            df['ASYM'] == asym].loc[
            df['GPOW'] == '_gpow_ave'+str(numsam)]
        else:
            reps = df.loc[df['NUMHM'] == nohm].loc[df['SNRPENALTY'] == snr_pen].loc[
            df['ASYM'] == asym].loc[
            df['GPOW'] == '_gpow_var'+str(numsam)]
    else:
        if samples:
            reps = df.loc[df['NUMHM'] == nohm].loc[df['SNRPENALTY'] == snr_pen].loc[
            df['ASYM'] == asym].loc[
            df['GPOW'] == '_gpow_av'+str(numsam)]
        else:
            reps = df.loc[df['NUMHM'] == nohm].loc[df['SNRPENALTY'] == snr_pen].loc[
            df['ASYM'] == asym].loc[
            df['GPOW'] == '_gpow_var'+str(numsam)]
    if param == 'ALPHA':
        return ((get_param_for_norm(reps, norm, param)/0.2 - 1.0)*100).to_numpy()
    elif param == 'GAMMA':
        return ((get_param_for_norm(reps, norm, param)/1.2 - 1.0)*100).to_numpy()
    elif param == 'NF':
        return ((get_param_for_norm(reps, norm, param)/4.5 - 1.0)*100).to_numpy()
    elif param == 'BTB':
        return ((get_param_for_norm(reps, norm, param)/14.8 - 1.0)*100).to_numpy()

def get_norm_for_plot(df, snr_pen, asym, nohm, numsam, norm, samples=True):
    if numsam == 1:  # required to deal with stupid naming convention...
        if samples:
            reps = df.loc[df['NUMHM'] == nohm].loc[df['SNRPENALTY'] == snr_pen].loc[
            df['ASYM'] == asym].loc[
            df['GPOW'] == '_gpow_ave'+str(numsam)]
        else:
            reps = df.loc[df['NUMHM'] == nohm].loc[df['SNRPENALTY'] == snr_pen].loc[
            df['ASYM'] == asym].loc[
            df['GPOW'] == '_gpow_var'+str(numsam)]
    else:
        if samples:
            reps = df.loc[df['NUMHM'] == nohm].loc[df['SNRPENALTY'] == snr_pen].loc[
            df['ASYM'] == asym].loc[
            df['GPOW'] == '_gpow_av'+str(numsam)]
        else:
            reps = df.loc[df['NUMHM'] == nohm].loc[df['SNRPENALTY'] == snr_pen].loc[
            df['ASYM'] == asym].loc[
            df['GPOW'] == '_gpow_var'+str(numsam)]
    return get_norm_val_for_norm(reps, norm).to_numpy()
def get_param_err_plot(sams, norm, df, snr_pen, param, asym, nohm, err_type, samples=True):
    """
    get error vector array for a given norm - assumes stability across 5 reps
    """
    err = np.zeros(sams.shape)
    for i in range(len(sams)):
        if err_type == 'ABSOLUTE':
            err[i] = get_error(df, snr_pen, param, asym, nohm, sams[i], norm, samples)[0]
        elif err_type == 'NORMALISED':
            err[i] = get_norm_error(df, snr_pen, param, asym, nohm, sams[i], norm, samples)[0]
    return err
def get_norm_plot(sams, norm, df, snr_pen, asym, nohm, err_type, samples=True):
    """
    get error vector array for a given norm - assumes stability across 5 reps
    """
    err = np.zeros(sams.shape)
    for i in range(len(sams)):
        err[i] = get_norm_for_plot(df, snr_pen, asym, nohm, sams[i], norm, samples)[0]
    return err
def print_results(inputs, predictions, truevals):
    "convenience function for printing out results and computing mean square error"

    print("Inputs                                   Pred. mean Actual Value")
    print("------------------------------------------------------------------------------")

    error = 0.

    for pp, m, trueval in zip(inputs, predictions, truevals):
        print("{}      {}       {}".format(np.around(pp,2), np.around(m,2), np.around(trueval,2)))
        error += (trueval - m)**2

    print("Mean squared error: {}".format(np.sqrt(error)/len(predictions)))
def get_validation_norms(gpe_direc, val_powers, offset_name, norm_type, gpe_dir):
    """
    get validation results for each launch power
    """
    norms = np.zeros([len(val_powers),1])
    for i in range(len(val_powers)):
        #val_pts = pickle.load(open("saved_models/"+gpe_dir+"/validation_points_ssfm"+val_pow+offset+".pkl", 'rb'))
        pred = pickle.load(open(gpe_dir+"/predictions_ssfm"+val_powers[i]+offset_name+".pkl", 'rb'))
        val_out = pickle.load(open(gpe_dir+"/validation_output_ssfm"+val_powers[i]+offset_name+".pkl", 'rb'))
        norms[i] = norm((pred.mean - val_out), norm_type)
    return norms
def get_full_intersec_res(data_direc, lop, pkp, hip, p4, p5, no_hm, run_name,
        offset_name, gpow_n, gpe_run, power_vals, powernames):
    _, _, _, _, _, _, intersec_res = generate_results(data_direc, lop, pkp, hip, p4, p5, no_hm, run_name,
                     offset_name, gpow_n, gpe_run, power_vals, powernames)
    return intersec_res
