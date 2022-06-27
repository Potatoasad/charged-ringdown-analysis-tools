from pylab import *

import arviz as az
import pandas as pd
import seaborn as sns
import multiprocessing

#multiprocessing.set_start_method("fork")

import ringdown
from exact_coeffs import interpolation_coeffs
#import ringdb



h_raw_strain = ringdown.Data.read('H-H1_GWOSC_16KHZ_R1-1126259447-32.hdf5', ifo='H1', kind='GWOSC')
l_raw_strain = ringdown.Data.read('L-L1_GWOSC_16KHZ_R1-1126259447-32.hdf5', ifo='L1', kind='GWOSC')

if __name__ == '__main__':
	x = linspace(0, 2, 9)
	for i in [1, 2, 4]:
	    plot(x[::i], exp(-x[::i]), label=r'$\tau / \Delta t = {:.0f}$'.format(4/i))
	legend(title='decimation')
	xlabel(r'$t / \tau$')
	ylabel(r'$\exp\left( - t / \tau \right)$');

	M_est = 70.0
	chi_est = 0.7

	longest_tau = ringdown.qnms.get_ftau(M_est, chi_est, 0, l=2, m=2)[1]
	highest_drate = 1/ringdown.qnms.get_ftau(M_est, chi_est, 1, l=2, m=2)[1]
	print('The damping rate of the second tone is: {:.1f} Hz'.format(highest_drate))
	print('The time constant of the first tone is: {:.1f} ms'.format(1000*longest_tau))

	def next_pow_two(x):
	    y = 1
	    while y < x:
	        y = y << 1
	    return y

	T = 10*longest_tau
	srate = next_pow_two(2*highest_drate)

	print('Segment of {:.1f} ms at sample rate {:.0f}'.format(1000*T, srate))

	fit = ringdown.Fit(model='mchiq_exact', modes=[(1, -2, 2, 2, 0), (1, -2, 2, 2, 1)],**interpolation_coeffs)  # use model='ftau' to fit damped sinusoids instead of +/x polarized GW modes
	fit.add_data(h_raw_strain)
	fit.add_data(l_raw_strain)
	fit.set_target(1126259462.4083147, ra=1.95, dec=-1.27, psi=0.82, duration=T)
	fit.condition_data(ds=int(round(h_raw_strain.fsamp/srate)), flow=1/T)

	fit.compute_acfs()
	wd = fit.whiten(fit.analysis_data)
	plot(wd['H1'], label='H1')
	plot(wd['L1'], label='L1')
	legend(loc='best');
	xlabel(r'Sample Number');
	ylabel(r'Whitened Analysis Data');

	print(fit.valid_model_options)
	fit.update_prior(A_scale=2e-21, M_min=35.0, M_max=140.0, flat_A=1)
	print(fit.prior_settings)
	fit.run(target_accept=0.95)

	az.to_netcdf(fit.result, 'GW150914.nc')


