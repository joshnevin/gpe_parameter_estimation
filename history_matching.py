import numpy as np
import mogp_emulator
import pickle
import argparse

parser = argparse.ArgumentParser(description='Set up simulation.')
parser.add_argument('--no_runs', default='5', type=int)
parser.add_argument('--npred', default='1e6', type=float)
parser.add_argument('--gpow', default='', type=str)
parser.add_argument('--offset', default='', type=str)
parser.add_argument('--gt', default='', type=str)
parser.add_argument('--gpe_run', default='', type=str)
args = parser.parse_args()

no_runs = args.no_runs
gpow_name = args.gpow
offset = args.offset
gt = args.gt
gpe_run = args.gpe_run

def hist_matching(power_name, n_predict, thresh):

    gp = pickle.load(open("saved_gpes/fitted_GPE_ssfm"+str(power_name)+offset+"_hipres"+gpe_run+".pkl", 'rb'))
    lhd = pickle.load(open("saved_gpes/lhd_ssfm_"+str(power_name)+offset+"_hipres"+gpe_run+".pkl", 'rb'))
    ground_truth = pickle.load(open("gt_data/ssfm_full_power_ground_truth_sim_hipres"+gt+gpow_name+".pkl", 'rb')).reshape(211,)

    prediction_points = lhd.sample(n_predict)

    if power_name == '-112':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[38]], threshold=thresh)
    elif power_name == '-103':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[47]], threshold=thresh)
    elif power_name == '-94':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[56]], threshold=thresh)
    elif power_name == '-93':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[57]], threshold=thresh)
    elif power_name == '-85':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[65]], threshold=thresh)
    elif power_name == '-82':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[68]], threshold=thresh)
    elif power_name == '-74':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[76]], threshold=thresh)
    elif power_name == '-70':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[80]], threshold=thresh)
    elif power_name == '-68':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[82]], threshold=thresh)
    elif power_name == '-62':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[88]], threshold=thresh)
    elif power_name == '-61':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[89]], threshold=thresh)
    elif power_name == '-51':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[99]], threshold=thresh)
    elif power_name == '-45':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[105]], threshold=thresh)
    elif power_name == '-11':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[139]], threshold=thresh)
    elif power_name == '18':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[168]], threshold=thresh)
    elif power_name == '17':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[167]], threshold=thresh)
    elif power_name == '28':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[178]], threshold=thresh)
    elif power_name == '27':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[177]], threshold=thresh)
    elif power_name == '35':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[185]], threshold=thresh)
    elif power_name == '41':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[191]], threshold=thresh)
    elif power_name == '40':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[190]], threshold=thresh)
    elif power_name == '50':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[200]], threshold=thresh)
    elif power_name == '49':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[199]], threshold=thresh)
    elif power_name == '-34':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[116]], threshold=thresh)
    elif power_name == '10':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[160]], threshold=thresh)
    elif power_name == '-27':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[123]], threshold=thresh)
    elif power_name == '5':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[155]], threshold=thresh)
    elif power_name == '-39':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[111]], threshold=thresh)
    elif power_name == '9':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[159]], threshold=thresh)
    elif power_name == '-31':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[119]], threshold=thresh)
    elif power_name == '4':
        hm = mogp_emulator.HistoryMatching(gp=gp, coords=prediction_points, obs=[ground_truth[154]], threshold=thresh)


    nroy_points = hm.get_NROY()
    print("Ruled out {} of {} points".format(n_predict - len(nroy_points), n_predict))
    return nroy_points, prediction_points

thresh = 3.0
n_predict = int(args.npred)

if offset == "_asymr":
    powers = ['-103','-85','-61','-45','-34','-11','10','18','27','40','49'] # right 0.01 dB SNR pres 3 meas
elif offset == "_asyml":
    powers = ['-112','-93','-68','-51','-39','-11','9','17','28','41','50'] # left 0.01 dB SNR pres 3 meas
for i in range(no_runs):
    for pow in powers:
        nroy, pdpts = hist_matching(pow, n_predict, thresh)
        hm_res = pdpts[nroy]
        if i == 0:  # for consistency with previous naming convention
            if len(gpow_name) == 0:
                if len(gpe_run) == 0:
                    pickle.dump(hm_res, open("hm_results/hm_results"+str(pow)+"_dbm_"+str(n_predict)+"_3sig_noD"+offset+gt+".pkl", 'wb'))
                else:
                    pickle.dump(hm_res, open("hm_results/hm_results"+str(pow)+"_dbm_"+str(n_predict)+"_3sig_noD"+offset+gt+"_gpe"+gpe_run+".pkl", 'wb'))
            else:
                if len(gpe_run) == 0:
                    pickle.dump(hm_res, open("hm_results_gpow/hm_results"+ str(pow) +"_dbm_"+str(n_predict)+"_3sig_noD"+gpow_name+offset+gt+".pkl", 'wb'))
                else:
                    pickle.dump(hm_res, open("hm_results_gpow/hm_results"+ str(pow) +"_dbm_"+str(n_predict)+"_3sig_noD"+gpow_name+offset+gt+"_gpe"+gpe_run+".pkl", 'wb'))
        else:
            if len(gpow_name) == 0:
                if len(gpe_run) == 0:
                    pickle.dump(hm_res, open("hm_results/hm_results"+ str(pow) +"_dbm_"+str(n_predict)+"_3sig_noD_run"+str(i+1)+offset+gt+".pkl", 'wb'))
                else:
                    pickle.dump(hm_res, open("hm_results/hm_results"+ str(pow) +"_dbm_"+str(n_predict)+"_3sig_noD_run"+str(i+1)+offset+gt+"_gpe"+gpe_run+".pkl", 'wb'))
            else:
                if len(gpe_run) == 0:
                    pickle.dump(hm_res, open("hm_results_gpow/hm_results"+ str(pow) +"_dbm_"+str(n_predict)+"_3sig_noD"+gpow_name+"_run"+str(i+1)+offset+gt+".pkl", 'wb'))
                else:
                    pickle.dump(hm_res, open("hm_results_gpow/hm_results"+ str(pow) +"_dbm_"+str(n_predict)+"_3sig_noD"+gpow_name+"_run"+str(i+1)+offset+gt+"_gpe"+gpe_run+".pkl", 'wb'))
