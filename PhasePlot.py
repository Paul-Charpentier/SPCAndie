## Imports

import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from tqdm import tqdm
from wpca import PCA, WPCA, EMPCA
from astropy.timeseries import LombScargle
from hampel import hampel
import pandas as pd
from scipy.stats.stats import pearsonr
from PyAstronomy.pyasl import binningx0dt
import sys
from astropy.stats import sigma_clip
import warnings
warnings.filterwarnings("ignore")
import matplotlib.animation as animation
from scipy import signal
from scipy.optimize import leastsq
frequency = np.linspace(1/1000, 1/1.1, 100000) # periodogram frequency grid

## Load data

path = '/media/paul/One Touch2/SPIRou_Data/0.7.275/EV_LAC/EV_LAC' #### PATH TO CHANGE ####
os.chdir(path)
ALL_d2v = []
ALL_sd2v = []
times = []
BERV = []
print('loading data...')
for (root, dirs, file) in os.walk(path):
    for f in tqdm(sorted(file)):
        if 'lbl.fits' in f:
            nthfile = fits.open(f, memmap=False)
            times.append(nthfile[0].header['BJD'])
            BERV.append(nthfile[0].header['BERV'])
            ALL_d2v.append(nthfile[1].data['d2v'])
            ALL_sd2v.append(nthfile[1].data['sd2v'])
            nthfile.close()

ALL_d2v = np.array(ALL_d2v)
ALL_sd2v = np.array(ALL_sd2v)
times = np.array(times)
BERV = np.array(BERV)

p1 = fits.open('/media/paul/One Touch2/SPIRou_Data/0.7.275/EV_LAC/EV_LAC/2438039o_pp_e2dsff_tcorr_AB_EV_LAC_EV_LAC_lbl.fits') #### PATH TO CHANGE ####
w1 = (p1[1].data['WAVE_START']+p1[1].data['WAVE_END'])/2.
Prot = 4.3715

## Remove some NaNs

print('NaNs Removal...')
def remove_nan(X, threshold = 200):

    used_waves = []
    for idx in range(X.shape[1]):
        if np.sum(np.isnan(X[:,idx]))<=threshold:
            used_waves.append(True)
        else:
            used_waves.append(False)

    return used_waves

used_waves = remove_nan(ALL_d2v, threshold = len(times)//2)
ALL_d2v = ALL_d2v[:, used_waves]
ALL_sd2v = ALL_sd2v[:, used_waves]
w_used = w1[used_waves]

## Night Binn

print('Night binning...')
def night_bin(times, rv, drv=None, binsize=0.5):
    """
    Bin data by night, using a moving window of size `binsize`.    Parameters
    ----------
    times : numpy.ndarray
        The time array.
    rv : numpy.ndarray
        The RV array.
    drv : numpy.ndarray, optional
        The error on the RV array. If provided, the weighted mean and variance of the RV values will be
        calculated. If not provided, the unweighted mean and variance will be calculated.
    binsize : float, optional
        The size of the moving window, in days.

    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
       The time, RV, and error on RV arrays for the binned data.
    """

    n = len(rv)
    res_times = np.empty(n)
    res_rv = np.empty(n)
    res_drv = np.empty(n)
    times_temp = []
    rv_temp = []
    drv_temp = []
    time_0 = times[0]
    res_index = 0
    for index in range(n):
        if np.abs(times[index] - time_0) <= binsize:
            times_temp.append(times[index])
            rv_temp.append(rv[index])
            if drv is not None:
                drv_temp.append(drv[index])
        else:
            times_temp = np.array(times_temp)
            res_times[res_index] = times_temp.mean()
            rv_temp = np.array(rv_temp)
            if drv is not None:
                drv_temp = np.array(drv_temp)
            mask = ~np.isnan(rv_temp)
            if mask.sum() > 0:
                if drv is not None:
                    weights = 1 / (drv_temp[mask]) ** 2
                    weights /= weights.sum()
                    average = (weights * rv_temp[mask]).sum()
                    var = (weights ** 2 * (drv_temp[mask]) ** 2).sum()
                    res_rv[res_index] = average
                    res_drv[res_index] = np.sqrt(var)
                else:
                    res_rv[res_index] = rv_temp[mask].mean()
                    res_drv[res_index] = rv_temp[mask].std()
                res_index += 1
            else:
                res_rv[res_index] = np.nan
                res_drv[res_index] = np.nan
                res_index += 1
            time_0 = times[index]
            times_temp = [time_0]
            rv_temp = [rv[index]]
            if drv is not None:
                drv_temp = [drv[index]]
    return res_times[:res_index], res_rv[:res_index], res_drv[:res_index]

tbinn, d2vbinn, sd2vbinn = [], [], []
for idx in tqdm(range(ALL_d2v.shape[1])):
    ttemp, d2vtemp, sd2vtemp  = night_bin(times, ALL_d2v[:,idx], ALL_sd2v[:,idx])
    d2vbinn.append(d2vtemp)
    sd2vbinn.append(sd2vtemp)
tbinn, d2vbinn, sd2vbinn = np.array(ttemp), np.array(d2vbinn).T, np.array(sd2vbinn).T

__, bervbin, __ = night_bin(times, BERV)

## Pre-Remove outliers by sigmaclipping

print('Outliers removal...')
d2vbinn = sigma_clip(d2vbinn, sigma=3, axis=0, masked=False)

## standardisation

ma_dv = np.ma.MaskedArray(d2vbinn.T, mask=np.isnan(d2vbinn.T))
ma_sdv = np.ma.MaskedArray(sd2vbinn.T, mask=np.isnan(sd2vbinn.T))
#Compute average
avg_dv = np.ma.average(ma_dv, weights=1/ma_sdv**2, axis=1)
std_dv = np.ma.average((ma_dv-avg_dv.reshape(-1,1))**2, weights=1/ma_sdv**2, axis=1)
#Reshape average
avg_dv = avg_dv.data.reshape(-1,1)
std_dv = np.sqrt(std_dv.data.reshape(-1,1))
#Normalize
RV2 = (np.copy(d2vbinn.T) - avg_dv)/std_dv
dRV2 = np.copy(sd2vbinn.T)/std_dv

## First PCA no correction


weights = 1. / dRV2
weights[np.isnan(RV2)] = 0
# Run pca
pca = WPCA(n_components=RV2.shape[1])
pca.regularization = 2
pca.fit(RV2, weights=weights)

fig, ax = plt.subplots(2, 3, figsize=(20, 5))
ax[0, 0].plot(bervbin, pca.components_[0], '.')
ax[0, 0].set_xlabel('Vtot (km/s)')
ax[0, 0].set_ylabel('W1')

ls = LombScargle(tbinn, pca.components_[0])#.autopower(minimum_frequency=0.0005, maximum_frequency=1/1.5) #nyquist_factor=15)
power = ls.power(frequency)
ax[0, 1].plot(1/frequency, power, 'r')
ax[0, 1].set_ylabel("power")
ax[0, 1].set_xscale('log')
fap = ls.false_alarm_level(0.1)
ax[0, 1].axhline(fap, linestyle='-', color='k')
fap = ls.false_alarm_level(0.01)
ax[0, 1].axhline(fap, linestyle='--', color='k')
fap = ls.false_alarm_level(0.001)
ax[0, 1].axhline(fap, linestyle=':', color='k')
ax[0, 1].axvline(Prot, linestyle=':', color='b', alpha=0.5)

ax[0, 1].set_title('No correction')

ax[0, 2].plot(tbinn, pca.components_[0], '.')
ax[0, 2].set_xlabel('BJD')
ax[0, 2].set_ylabel('W1')

ax[1, 0].plot(bervbin, pca.components_[1], '.')
ax[1, 0].set_xlabel('Vtot (km/s)')
ax[1, 0].set_ylabel('W2')

ls = LombScargle(tbinn, pca.components_[1])#.autopower(minimum_frequency=0.0005, maximum_frequency=1/1.5) #nyquist_factor=15)
power = ls.power(frequency)
ax[1, 1].plot(1/frequency, power, 'r')
ax[1, 1].set_ylabel("power")
ax[1, 1].set_xscale('log')
fap = ls.false_alarm_level(0.1)
ax[1, 1].axhline(fap, linestyle='-', color='k')
fap = ls.false_alarm_level(0.01)
ax[1, 1].axhline(fap, linestyle='--', color='k')
fap = ls.false_alarm_level(0.001)
ax[1, 1].axhline(fap, linestyle=':', color='k')
ax[1, 1].axvline(Prot, linestyle=':', color='b', alpha=0.5)


ax[1, 2].plot(tbinn, pca.components_[1], '.')
ax[1, 2].set_xlabel('BJD')
ax[1, 2].set_ylabel('W2')
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/Gl410/BERV_Before.png')
plt.show()

## Fit 1yr signal function

def popnan(ind):
    RVout = RV2[ind, np.invert(np.isnan(RV2[ind]))]
    tout = tbinn[np.invert(np.isnan(RV2[ind]))]
    dRVout = dRV2[ind, np.invert(np.isnan(RV2[ind]))]
    return(tout, RVout, dRVout)

def fityr(ind, display=False):

    t, data, ddata = popnan(ind)

    # first guess
    guess_amp = np.nanstd(data)
    guess_phase = 0
    guess_mean = np.nanmean(data)

    if display:
        tlin = np.linspace(np.min(t), np.max(t), 1000)
        plt.plot(t, data, '.')
        plt.plot(tlin, guess_amp*np.sin((2*np.pi/4.3715)*tlin+guess_phase) + guess_mean, 'r-')

    #fit
    optimize_func = lambda x: x[0]*np.sin((2*np.pi/4.3715)*t+x[1]) + x[2] - data
    est_amp, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_phase, guess_mean])[0]

    datasim = est_amp*np.sin((2*np.pi/4.3715)*t+est_phase) + est_mean

    corr = pearsonr(data, datasim)[0]

    if display:
        plt.plot(tlin, est_amp*np.sin((2*np.pi/4.3715)*tlin+est_phase) + est_mean, 'g-')
        plt.show()

    return(est_amp, est_phase, corr)

## Polar periodo

AMP, PHASE, CORR = [],[],[]
w_filtr = []
for l in tqdm(range(len(RV2))):
    a, p, c= fityr(l)
    if a < 0:
        a = -a
        p = p+np.pi
    if p>0.08*np.pi and p<np.pi/4:
        w_filtr.append(True)
    else:
        w_filtr.append(False)
    AMP.append(a)
    PHASE.append(p)
    CORR.append(c)

fig, ax = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
ax[0].scatter(PHASE, AMP, cmap="rainbow", c=np.array(CORR), s=np.array(CORR)*21)
ax[0].set_title('Amplitude')
ax[1].scatter(PHASE, CORR, cmap="rainbow", c=np.array(AMP), s=np.array(AMP)*21)
ax[1].set_title('Correlation')
plt.show()

## Sort

RV2_filtr = RV2[w_filtr]
dRV2_filtr = dRV2[w_filtr]

## PCA after filtering


weights = 1. / dRV2_filtr
weights[np.isnan(RV2_filtr)] = 0
# Run pca
pca = WPCA(n_components=RV2_filtr.shape[1])
pca.regularization = 2
pca.fit(RV2_filtr, weights=weights)

fig, ax = plt.subplots(2, 3, figsize=(20, 5))
ax[0, 0].plot(bervbin, pca.components_[0], '.')
ax[0, 0].set_xlabel('Vtot (km/s)')
ax[0, 0].set_ylabel('W1')

frequency, power = LombScargle(tbinn, pca.components_[0]).autopower(minimum_frequency=0.0005, maximum_frequency=1/1.5) #nyquist_factor=15)
ax[0, 1].plot(1/frequency, power, 'r')
ax[0, 1].set_ylabel("power")
ax[0, 1].set_xscale('log')
ls = LombScargle(tbinn, pca.components_[0])
fap = ls.false_alarm_level(0.1)
ax[0, 1].axhline(fap, linestyle='-', color='k')
fap = ls.false_alarm_level(0.01)
ax[0, 1].axhline(fap, linestyle='--', color='k')
fap = ls.false_alarm_level(0.001)
ax[0, 1].axhline(fap, linestyle=':', color='k')
ax[0, 1].axvline(4.3715, linestyle=':', color='b', alpha=0.5)

ax[0, 1].set_title('phaseplot filtered')

ax[0, 2].plot(tbinn, pca.components_[0], '.')
ax[0, 2].set_xlabel('BJD')
ax[0, 2].set_ylabel('W1')

ax[1, 0].plot(bervbin, pca.components_[1], '.')
ax[1, 0].set_xlabel('Vtot (km/s)')
ax[1, 0].set_ylabel('W2')

frequency, power = LombScargle(tbinn, pca.components_[1]).autopower(minimum_frequency=0.0005, maximum_frequency=1/1.5) #nyquist_factor=15)
ax[1, 1].plot(1/frequency, power, 'r')
ax[1, 1].set_ylabel("power")
ax[1, 1].set_xscale('log')
ls = LombScargle(tbinn, pca.components_[1])
fap = ls.false_alarm_level(0.1)
ax[1, 1].axhline(fap, linestyle='-', color='k')
fap = ls.false_alarm_level(0.01)
ax[1, 1].axhline(fap, linestyle='--', color='k')
fap = ls.false_alarm_level(0.001)
ax[1, 1].axhline(fap, linestyle=':', color='k')
ax[1, 1].axvline(4.3715, linestyle=':', color='b', alpha=0.5)


ax[1, 2].plot(tbinn, pca.components_[1], '.')
ax[1, 2].set_xlabel('BJD')
ax[1, 2].set_ylabel('W2')
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/Gl410/PacMap_filtred.png')
plt.show()
#
# ## Make Gif
#
# def yarara(Y, display=True):
#     AMP, PHASE = [],[]
#     w_filtr = []
#     for l in range(len(RV2)):
#         a, p = fityr(l)
#         if a < 0:
#             a = -a
#             p = p+np.pi
#         if np.abs(a)>Y:
#             w_filtr.append(False)
#         else:
#             w_filtr.append(True)
#         AMP.append(a)
#         PHASE.append(p)
#     # Sort
#     RV2_filtr = RV2[w_filtr]
#     dRV2_filtr = dRV2[w_filtr]
#     # PCA after filtering
#     weights = 1. / dRV2_filtr
#     weights[np.isnan(RV2_filtr)] = 0
#     # Run pca
#     pca = WPCA(n_components=RV2_filtr.shape[1])
#     pca.regularization = 2
#     pca.fit(RV2_filtr, weights=weights)
#     #Plot
#
#     ls = LombScargle(tbinn, pca.components_[0])
#     power = ls.power(frequency)
#     fap = ls.false_alarm_probability(power.max())
#     if display:
#         plt.clf()
#         plt.plot(1/frequency, power, 'k-')
#     fap =ls.false_alarm_probability(power.max())
#     # false alarm levels
#     fap = ls.false_alarm_level(1e-3)
#     if display:
#         plt.axhline(y=fap,linestyle="-",color="k")
#     fap = ls.false_alarm_level(1e-5)
#     if display:
#         plt.axhline(y=fap,linestyle="--",color="k")
#         plt.xscale('log')
#         plt.annotate('Y = ' + str(Y), (110, 0.225))
#         plt.annotate('#  = ' + str(RV2_filtr.shape[0]), (110, 0.25))
#     return(RV2_filtr.shape[0], 1/frequency[np.argmax(power)], np.max(power))
#
# ## Gif
# #
# fig = plt.figure(figsize=(16, 9))
# WOAWANIMTROBELLE = animation.FuncAnimation(fig, yarara, frames = np.arange(3, 0, -0.1))#np.arange(len(times_tcorr)))
# WOAWANIMTROBELLE.save('/home/paul/Bureau/IRAP/dLWPCA/Yscore.gif', writer=animation.PillowWriter(fps=4))
# plt.show()
#
# ## Log fap evolution
#
# LOGFAP = []
# MAXFRE = []
# NBRLIN = []
# for z in tqdm(np.arange(3, 0, -0.1)):
#     impacted, maxfreq, logfap = yarara(z, display=False)
#     MAXFRE.append(maxfreq)
#     LOGFAP.append(logfap)
#     NBRLIN.append(impacted)
#
#
# fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
# ax[0].plot(3-np.arange(3, 0, -0.1), LOGFAP)
# ax[0].set_ylabel('max power')
#
# ax[1].plot(3-np.arange(3, 0, -0.1), MAXFRE)
# ax[1].set_ylabel('frequency')
#
# ax[2].plot(3-np.arange(3, 0, -0.1), NBRLIN)
# ax[2].set_ylabel('# lines')
# ax[2].set_xlabel('3-y0')
# plt.show()