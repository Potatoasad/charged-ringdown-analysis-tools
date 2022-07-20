import configparser
import argparse

from sys import exit
import os
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Override parameter arguments')
parser.add_argument('-e','--eventname',type=str, nargs='?',help='name of the event')
parser.add_argument('-c','--configfile', type=str, nargs='?', default="config.ini",help='path to config file')
parser.add_argument('--charged',const=True, default=None, action='store_const',help='use --charged to make it a charged analysis')
parser.add_argument('--foldersonly',const=True, default=False, action='store_const',help='use --foldersonly to only make the necessary folders')
parser.add_argument('--diagnosticsonly',const=True, default=False, action='store_const',help='use --diagnosticssonly to only make the necessary diagnostic plots')
parser.add_argument('--skipdiagnostics',const=True, default=False, action='store_const',help='use --skipdiagnostics to skip diagnostic plots')
parser.add_argument('--exactcharge',const=True, default=False, action='store_const',help='use --exactcharge to run the exact charge formula')

args = parser.parse_args()

config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read(getattr(args, "configfile"))

#Parsing
params = {}
for section in config.sections():
	for key in config[section].keys():
		dtype = 'str' # default type

		# Pick out the type from key
		if "|" in key:
			newkey, dtype = key.split(" | ")
			dtype = dtype.strip()
		else:
			newkey = key

		# Convert types if needed
		if dtype in ['bool', 'float', 'eval']:
			value = eval(config[section][key].strip())
		else:
			value = config[section][key]

		# Use the default value from CLI arguments
		params[newkey] = getattr(args, newkey, None) if getattr(args, newkey, None) is not None else value

# Model
eventname = params['eventname']
#print(f"{eventname}")
print(f"Run for the event: {eventname}")
charged = params['charged']
exactcharge = args.exactcharge

# Files and Folders
dbfolder = params["databasefolder"]

# Make the parent results folder if its not there
parent_result = "/".join(params["resultsfolder"].split("/")[0:-1])
if not os.path.exists(parent_result):
	subprocess.run(["mkdir", parent_result])

# Make the event's result folder if its not there
resultsfolder = params["resultsfolder"]
if not os.path.exists(resultsfolder):
	subprocess.run(["mkdir", resultsfolder])

# Make the plotting folder if not there
plotting = params['plotting']
if plotting and not os.path.exists(params["plotsfolder"]):
	subprocess.run(["mkdir", params["plotsfolder"]])

# If you just wanted to make folders stop 
# the process here
if args.foldersonly:
	exit()

#================================================================
# Now we can start the analysis in earnest
#================================================================

# Define the database
import ringdown
from ringdb import Database, PSD
from ringdown import IMR

db = Database(dbfolder)
db.initialize()

event = db.event(eventname)

# Get the strain data
strain = event.strain()

# Get the posteriors
posteriors = event.posteriors()

'''# Get the PSD data
psds = event.psd()

# Replace PSD with a newer custom CSV if provided
#print("Using a custom PSD?: ", params["use_custom_psd"])
if params["use_custom_psd"]:
	delim = params["psd_delim"]
	psds = {}
	for ifo in ['H1','L1','V1']:
		psd_path = params[f"{ifo.lower()}_psd_path"]
		if len(psd_path) != 0:
			header = 0 if params["psd_header_exists"] else None
			col_names = params["psd_column_names"]
			psd_df = pd.read_csv(psd_path, delimiter=delim, header=header, names=col_names)
			psds[ifo] = PSD(psd_df['power'].values, index=psd_df['freqs'].values)


# Some PSDs have an np.inf near the edges
clear_infinities = lambda x: x.replace([np.inf, -np.inf], np.nan).fillna(method='bfill').fillna(method='ffill')
psds = {ifo: clear_infinities(psd) for ifo, psd in psds.items()}

# PSD Processing
new_psds = {}
for ifo, psd in psds.items():
	from_freq = max(psd.freq.min(), params["psd_low_freq"])
	thepsd = psd.low_pad(from_freq=from_freq).high_pad()
	if params["interpolate"]:
		minfreq = thepsd.freq.min()
		maxfreq = thepsd.freq.max()
		if ((np.log(maxfreq)/np.log(2)) % 1) != 0:
			lowest_2_pow = 2**np.floor((np.log(maxfreq)/np.log(2)))
			thepsd = thepsd[minfreq:lowest_2_pow]
			thepsd = thepsd.interpolate()
	new_psds[ifo] = thepsd


# Get a list of good detectors we can use:
# is that ifo in the strain?
ifos = list(strain.keys())
# does that ifo have a usable PSD??
usable_psd_ifos = []
for ifo in ifos:
	try:
		new_psds[ifo].to_acf().cholesky
		usable_psd_ifos.append(ifo)
	except:
		print(f"PSD from ifo: {ifo} is not usable")

usable_ifos = list(filter(lambda ifo: ifo in usable_psd_ifos, ifos))
print(usable_ifos)

#Set all things to just contain the new ifos, that is it:
strain = {ifo: data for ifo, data in strain.items() if ifo in usable_ifos}
new_psds = {ifo: data for ifo, data in new_psds.items() if ifo in usable_ifos}
acfs = {ifo: data.to_acf() for ifo, data in new_psds.items() if ifo in usable_ifos}
'''

# Get the ifo that will be considered the 'main' ifo
possible_snr_from_posterior0 = [colname for colname in posteriors.columns]
possible_snr_from_posterior1 = [colname for colname in possible_snr_from_posterior0 if ('_matched_filter_snr_abs' in colname)]
possible_snr_from_posterior = [colname for colname in possible_snr_from_posterior1 if colname.replace('_matched_filter_snr_abs','') in usable_ifos]
print(possible_snr_from_posterior)
if len(possible_snr_from_posterior) == 0:
	print(possible_snr_from_posterior1)
	main_ifo = list(strain.keys())[0]
else:
	main_ifo = posteriors[possible_snr_from_posterior].mean().idxmax().replace('_matched_filter_snr_abs', '')

# Get the percentile t_peak sample
percentile = params["start_time_percentile"]/100
df_times = posteriors[posteriors.t_peak != 0.0].sort_values(f"{main_ifo}_peak", ascending=True)
peak_sample = df_times[df_times[f"{main_ifo}_peak"] <= df_times[f"{main_ifo}_peak"].quantile(percentile)].tail(1).squeeze().to_dict()


#================================================================
# Data Conditioning parameters:
#================================================================

# Data conditioning parameters
duration = params['duration']
fsamp = 2*new_psds[main_ifo].freq.max()
ds = int(np.round(strain[main_ifo].fsamp/fsamp))
flow = max(params['data_low_freq'], psds[main_ifo].freq.min())

# Find IMR reference frequencies
if not event.PD_ref.in_GWTC1(event.name):
	db.update_posterior_schema({'f_low': {'path': "/{approximant}/meta_data/meta_data/f_low", 'type':'value'}})
	db.update_posterior_schema({'f_ref': {'path': "/{approximant}/meta_data/meta_data/f_ref", 'type':'value'}})

	f_low = float(event.read_posterior_file_from_schema('f_low'))
	f_ref = float(event.read_posterior_file_from_schema('f_ref'))
else:
	f_low = 20.0
	f_ref = 20.0

# Create an IMR fromt he posterior
imr = IMR.from_posterior(peak_sample, f_low=f_low, f_ref=f_ref)

print("imr.t0 = ", imr.t0)
print("peak median = ", peak_sample['t_peak'])
# Adjust the start time to contain the peak sample
start_time = imr.t0 - (1/fsamp)

for ifo in usable_ifos:
	print(f"acf {ifo} fsamp: ", acfs[ifo].fsamp)
	print(f"data {ifo} fsamp: ", strain[ifo].fsamp)
	print(f"psds {ifo} fsamp: ", psds[ifo].fsamp)
	print(f"new psds {ifo} fsamp: ", new_psds[ifo].fsamp)

# Lets make sure all ifos are available in strain and acfs
# and ignore V1s
"""
for ifo in ['H1','L1','V1']:
	if ((ifo not in strain.keys()) and (ifo in acfs.keys())) or (ifo in ['V1']):
		acfs.pop(ifo,None)
		psds.pop(ifo,None)
		new_psds.pop(ifo,None) 
	if ((ifo in strain.keys()) and (ifo not in acfs.keys())) or (ifo in ['V1']):
		strain.pop(ifo,None)

"""

#================================================================
# Diagnostic Plots:
#================================================================
#1. Plot the whitened IMR with the current conditioning parameters
#2. Plot the PSD with a generated ringdown signal in the fourier domain
#___________________________________________________________________


if not getattr(args, 'skipdiagnostics', False):
	# Turn off interactive mode
	plt.ioff()

	# Define SNR calculations
	def SNR_calculation(imr=imr, data=strain, acfs=acfs, t0=imr.t0, duration=duration, flow=from_freq):
		whitened_signal, whitened_data = imr.whiten_with_data(data=data,acfs=acfs,t0=t0,duration=duration,flow=flow)
		SNR_squared = 0.0
		SNRs = {}
		for ifo in whitened_data.keys():
		    signal_optimal_SNR_squared = np.dot(whitened_signal[ifo].values, whitened_signal[ifo].values)
		    calc_snr_squared = (np.dot(whitened_signal[ifo].values, whitened_data[ifo].values)**2)/signal_optimal_SNR_squared
		    SNR_squared += calc_snr_squared
		    SNRs[ifo] = np.sqrt(calc_snr_squared)
		SNRs['Total'] = np.sqrt(SNR_squared)
		return SNRs

	whitened_signal, whitened_data = imr.whiten_with_data(data=strain, acfs=acfs, t0=start_time, duration=duration, flow=from_freq)
	snrs = SNR_calculation(imr=imr, data=strain, acfs=acfs, t0=start_time, duration=duration, flow=from_freq)

	fig_imr_plot,axes = plt.subplots(len(whitened_data.keys()) + 2, figsize=(15,20))
	plt.suptitle(f"IMR Ringdown vs Data comparison for {eventname}")

	# IMR Comparison Plots
	for i,ifo in enumerate(whitened_data.keys()):
	    data = whitened_data[ifo]
	    signal_ifo = whitened_signal[ifo]
	    axes[i].errorbar(data.time, data.values, yerr=np.ones_like(data.values), fmt='.', 
	     alpha=0.5, label='Data')
	    axes[i].set_title(f"{ifo} IMR vs Data | SNR = {snrs[ifo]}")
	    axes[i].plot(signal_ifo.time, signal_ifo.values, color="black", label='IMR')
	    axes[i].set_xlabel(r'$t / \mathrm{s}$')
	    axes[i].set_ylabel(r'$h_%s(t)$ (whitened)' % ifo[0])

	    # Add t_peak if needed
	    if (data.time.min() < imr.t_dict[ifo]) and (imr.t_dict[ifo] < data.time.max()):
	        axes[i].axvline(imr.t_dict[ifo], linestyle='dashed', c='r', label='Peak Time', alpha=0.5)

	    axes[i].legend(loc='best')


	# SNR Plot
	duration_step = 0.0025 
	durations = np.arange(duration_step*2,0.12,duration_step)
	snrs = np.zeros_like(durations)
	for i,new_duration in enumerate(durations):
		snrs[i] = SNR_calculation(imr=imr, data=strain, acfs=acfs, t0=start_time, duration=new_duration, flow=from_freq)['Total']

	t0_90_low = float(df_times['t_peak'].quantile(0.05))
	t0_90_high = float(df_times['t_peak'].quantile(0.95))
	print(t0_90_low, t0_90_high)
	start_times = np.arange(t0_90_low,t0_90_high,0.0005)
	snrs_starttime = np.zeros_like(start_times)
	print([a.time.min() for ifo, a in strain.items()])
	print(imr.time.min())
	print("start time = ", start_time)
	for i,new_start_time in enumerate(start_times):
		snrs_starttime[i] = SNR_calculation(imr=imr, data=strain, acfs=acfs, t0=new_start_time, duration=duration, flow=from_freq)['Total']

	axes[-2].plot(durations, snrs)
	axes[-2].scatter(durations, snrs)
	axes[-2].set_xlabel("duration (s)")
	axes[-2].set_ylabel("Optimal Matched \n Filter SNR")

	axes[-1].plot(start_times, snrs_starttime)
	axes[-1].scatter(start_times, snrs_starttime)
	axes[-1].set_xlim([t0_90_low-duration_step*3,t0_90_high + duration_step*3])
	print(start_time)
	axes[-1].axvline(start_time, linestyle='dashed', c='r', label='Peak Time', alpha=0.5)
	ax2 = axes[-1].twinx()
	ax2.hist(df_times['t_peak'].values, bins=25, alpha=0.4, linewidth=0.0)
	axes[-1].axvline(df_times['t_peak'].median() - 1/fsamp, linestyle='dashed', c='b', label='Median Peak Time', alpha=0.5)
	axes[-1].set_xlabel("start time (s)")
	axes[-1].set_ylabel("Optimal Matched \n Filter SNR")
	axes[-1].legend(loc='best')
	ax2.set_ylabel("Frequency")

	fig_imr_plot.savefig(params['diganostic_imr_plot_file'])

	## PSD Plot with sample waveform on top
	#print("Making PSD diagnostic plot")
	M_est = peak_sample['final_mass']
	chi_est = peak_sample['final_spin']

	f0, tau0 = ringdown.qnms.get_ftau(M_est, chi_est, 0)
	f1, tau1 = ringdown.qnms.get_ftau(M_est, chi_est, 1)

	ts = imr[imr.t0:(imr.t0+0.1)].time

	wfs = {}

	k0 = r'$h_0$'
	k1 = r'$h_1$'
	k2 = r'$h$'

	wfs[k0] = imr.abs().max()*np.exp(-(ts-ts[0])/tau0)*np.cos(2*np.pi*f0*(ts-ts[0]))
	wfs[k1] = imr.abs().max()*np.exp(-(ts-ts[0])/tau1)*np.cos(2*np.pi*f1*(ts-ts[0]))
	wfs[k2] = wfs[k0] - 2*wfs[k1] # Approximate phase and amplitude relationship in the actual waveform...

	ring_est = ringdown.injection.IMR(wfs[k2], posterior_sample=peak_sample ,index=ts, t_dict=imr.t_dict)

	fig_diagnostic_psd_plot, axes = plt.subplots(len(new_psds.keys()), figsize=(10,10))
	plt.suptitle(f"PSD diagnostic plots for {eventname}")

	for i, ifo in enumerate(new_psds.keys()):
		psd = new_psds[ifo]
		old_psd = psds[ifo]
		ring_est_psd = getattr(ring_est, ifo).get_psd().interpolate(freqs=psd.freq)
		axes[i].loglog(psd, c='r' ,label="Conditioned")
		axes[i].loglog(old_psd, c='r', linestyle='--', label='Original')
		axes[i].loglog(ring_est_psd, c='b', linestyle='-.', label='Signal (estimated)')
		axes[i].set_ylim([ring_est_psd.min()/10, psd.max()*100])
		axes[i].set_title(f"PSD plot for {ifo}")
		axes[i].legend(loc='best')


	fig_diagnostic_psd_plot.savefig(params["diganostic_psd_plot_file"])

if getattr(args, 'diagnosticsonly', False):
	exit()


#================================================================
# Set the Conditioning variables
#================================================================

# duration, fsamp, ds and start_time have already been set

modes = [(x['p'],x['s'],x['l'],x['m'],x['n']) for x in params["modes"]]
df_coeffs = None
dg_coeffs = None


#================================================================
# Enough Diagnostics, now lets ACTUALLY run the analysis
#================================================================

import multiprocessing
#multiprocessing.set_start_method("fork")

if charged:
	shift_coeffs_df = pd.read_csv(params["charge_coeff_path"])

	df_coeffs = []; dg_coeffs = [];
	for i,x in shift_coeffs_df.iterrows():
		mode_shift = (1,-2,2,2,x['n'])
		if mode_shift in modes:
			df_coeffs.append(eval(x['df_coeffs']))
			dg_coeffs.append(eval(x['dg_coeffs']))

if not charged:
	fit = ringdown.Fit(model='mchi', modes=modes)
else:
	if exactcharge:
		from exact_coeffs import interpolation_coeffs
		fit = ringdown.Fit(model='mchiq_exact', modes=modes, **interpolation_coeffs)
	else:
		fit = ringdown.Fit(model='mchiq', modes=modes, df_coeffs=df_coeffs, dg_coeffs=dg_coeffs)

if charged and ('charged' not in params['resultsfile']):
	result_file = params['resultsfile'].replace(f"{event.name}-posteriors", f"{event.name}-charged-posteriors")
else:
	result_file = params['resultsfile']

print(result_file)

for ifo in usable_ifos:
	fit.add_data(strain[ifo])
fit.set_target(start_time, ra=peak_sample['ra'], dec=peak_sample['dec'], psi=peak_sample['psi'], duration=duration)
fit.condition_data(ds=ds, flow=flow)
fit.acfs = acfs
fit.update_prior(A_scale=5e-21, M_min=peak_sample['final_mass']*0.5, M_max=1.5*peak_sample['final_mass'])

print(fit.prior_settings)

if __name__ == '__main__':
	fit.run(draws=1000, target_accept=0.95)
	fit.result.to_netcdf(result_file)




"""
# Exact charge interpolation coeffs:
b_omega_0 = [[1.0, 0.537583, -2.990402, 1.503421],
	 [-1.899567, -2.128633, 6.626680, -2.903790],
	 [1.015454, 2.147094, -4.672847, 1.891731],
	 [-0.111430, -0.581706, 1.021061, -0.414517]]

c_omega_0 = [[1.0, 0.548651, -3.141145, 1.636377],
			[-2.238461, -2.291933, 7.695570, -3.458474],
			[1.581677, 2.662938, -6.256090, 2.494264],
			[-0.341455, -0.930069, 1.688288, -0.612643]]

b_omega_1 = [[1.0, -2.918987, 2.866252, -0.944554],
			 [-1.850299, 7.321955, -8.783456, 3.292966],
			 [0.944088, -5.584876, 7.675096, -3.039132],
			 [-0.088458, 1.198758, -1.973222, 0.838109]]

c_omega_1 = [[1.0,-2.941138, 2.907859, -0.964407],
			 [-2.250169, 8.425183, -9.852886, 3.660289],
			 [1.611393, -7.869432, 9.999751, -3.737205],
			 [-0.359285, 2.392321, -3.154979, 1.129776]]


Y0_omega = [0.37367168, 0.34671099]

b_gamma_0 = [[1.0, -2.721789, 2.472860, -0.750015],
				[-2.533958, 7.181110, -6.870324, 2.214689],
				[2.102750, -6.317887, 6.206452, -1.980749],
				[-0.568636, 1.857404, -1.820547, 0.554722]]

c_gamma_0 = [[1.0,-2.732346, 2.495049, -0.761581],
				[-2.498341, 7.089542, -6.781334, 2.181880],
				[2.056918, -6.149334, 6.010021, -1.909275],
				[-0.557557, 1.786783, -1.734461, 0.524997]]

b_gamma_1 = [[1.0, -3.074983, 3.182195, -1.105297],
			 [0.366066, 4.296285, -9.700146, 5.016955],
			  [-3.290350, -0.844265, 9.999863, -5.818349],
			  [1.927196, -0.401520, -3.537667, 2.077991]]

c_gamma_1 = [[1.0, -3.079686, 3.191889, -1.110140],
			 [0.388928, 4.159242, -9.474149, 4.904881],
			 [-3.119527, -0.914668, 9.767356, -5.690517],
			 [1.746957, -0.240680, -3.505359, 2.049254]]

Y0_gamma = [0.08896232, 0.27391488]

interpolation_coeffs = {'b_omega': [b_omega_0, b_omega_1],
						'c_omega': [c_omega_0, c_omega_1],
						'b_gamma': [b_gamma_0, b_gamma_1],
						'c_gamma': [c_gamma_0, c_gamma_1],
						'Y0_omega': Y0_omega,
						'Y0_gamma': Y0_gamma}

"""
