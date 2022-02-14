import numpy as np
import mogp_emulator
import pickle
import matlab.engine
import argparse
import os
import scipy
from scipy.stats import norm, randint

parser = argparse.ArgumentParser(description='Set up simulation.')
parser.add_argument('--power', default='-1.0', type=float)
parser.add_argument('--disp', default='17.0', type=float)
parser.add_argument('--n_sims', default=200, type=int)
parser.add_argument('--n_val', default=20, type=int)
parser.add_argument('--offset', default='', type=str)
parser.add_argument('--gpe_run', default='', type=str)
args = parser.parse_args()

# set-up latin hypercube design
# INPUTS: launch power, loss, dispersion, gamma, NF
power_value = args.power
disp_value = args.disp
offset = args.offset
gpe_run = args.gpe_run
if offset == '_asyml':
    lhd = mogp_emulator.LatinHypercubeDesign([(0.18, 0.21), (0.9, 1.4), (4.2, 4.7), (14.4, 15.1)])
elif offset == '_asymr':
    lhd = mogp_emulator.LatinHypercubeDesign([(0.19, 0.22), (1.0, 1.5), (4.3, 4.8), (14.5, 15.2)])
pickle.dump(lhd, open("saved_gpes/lhd_ssfm_"+str(round(10*power_value))+offset+"_hipres"+gpe_run+".pkl", 'wb'))  # save LHC design

n_simulations = args.n_sims
simulation_points = lhd.sample(n_simulations)
pickle.dump(simulation_points, open("saved_gpes/simulation_points_ssfm"+str(round(10*power_value))+offset+"_hipres"+gpe_run+".pkl", 'wb'))  # save LHC design

print("Drawing simulated points from the experimental design:")
print("Starting MATLAB engine...")
eng = matlab.engine.start_matlab()
cwd = os.getcwd()
path = cwd + '/SSF'
eng.addpath(path,nargout=0)
simulation_output = []
for i in range(n_simulations):
    simulation_output.append(eng.SSFM_f(power_value, simulation_points[i][0].item(),
    disp_value, simulation_points[i][1].item(), simulation_points[i][2].item(),
    simulation_points[i][3].item(), nargout=1))
    print("Completed iter " + str(i+1) + " of " + str(n_simulations))
eng.quit()
print("Quit MATLAB engine.")

simulation_output = np.array(simulation_output)
pickle.dump(simulation_output, open("saved_gpes/simulation_output_ssfm"+str(round(10*power_value))+offset+"_hipres"+gpe_run+".pkl", 'wb')) # save simulation outputs

# Next, fit the surrogate GP model using MLE (MAP with uniform priors)
# Print out hyperparameter values as correlation lengths and sigma
print("Fitting surrogate GP model using MLE:")
gp = mogp_emulator.GaussianProcess(simulation_points, simulation_output)
gp = mogp_emulator.fit_GP_MAP(gp)

pickle.dump(gp, open("saved_gpes/fitted_GPE_ssfm"+str(round(10*power_value))+offset+"_hipres"+gpe_run+".pkl", 'wb'))

print("Correlation lengths = {}".format(np.sqrt(np.exp(-gp.theta[:2]))))
print("Sigma = {}".format(np.sqrt(np.exp(gp.theta[2]))))

# Validate emulator by comparing to true simulated value
# To compare with the emulator, use the predict method to get mean and variance
# values for the emulator predictions and see how many are within 2 standard
# deviations

n_valid = args.n_val
validation_points = lhd.sample(n_valid)
pickle.dump(validation_points, open("saved_gpes/validation_points_ssfm"+str(round(10*power_value))+offset+"_hipres"+gpe_run+".pkl", 'wb')) # save validation points

print("Drawing validation points from experimental design:")
print("Starting MATLAB engine...")
eng = matlab.engine.start_matlab()
cwd = os.getcwd()
path = cwd + '/SSF'
eng.addpath(path,nargout=0)
validation_output = []
for i in range(n_valid):
    validation_output.append(eng.SSFM_f(power_value, validation_points[i][0].item(),
    disp_value, validation_points[i][1].item(), validation_points[i][2].item(),
    validation_points[i][3].item(), nargout=1))
    print("Completed iter " + str(i+1) + " of " + str(n_valid))
eng.quit()
print("Quit MATLAB engine.")

validation_output = np.array(validation_output)
pickle.dump(validation_output, open("saved_gpes/validation_output_ssfm"+str(round(10*power_value))+offset+"_hipres"+gpe_run+".pkl", 'wb')) # save validation output

print("Performing validation:")
predictions = gp.predict(validation_points)
pickle.dump(predictions, open("saved_gpes/predictions_ssfm"+str(round(10*power_value))+offset+"_hipres"+gpe_run+".pkl", 'wb')) # save prediction values
