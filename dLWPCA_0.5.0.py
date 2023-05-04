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

## Load Data

path = '/media/paul/One Touch2/SPIRou_Data/0.7.275/EV_LAC/EV_LAC' #### PATH TO CHANGE ####
os.chdir(path)
ALL_d2v = []
ALL_sd2v  = []
ALL_dv = []
ALL_sdv = []
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
            ALL_dv.append(nthfile[1].data['dv'])
            ALL_sdv.append(nthfile[1].data['sdv'])
            nthfile.close()

ALL_d2v  = np.array(ALL_d2v)
ALL_sd2v = np.array(ALL_sd2v)
ALL_dv = np.array(ALL_dv)
ALL_sdv = np.array(ALL_sdv)
times = np.array(times)
BERV = np.array(BERV)

p1 = fits.open('/media/paul/One Touch2/SPIRou_Data/0.7.275/EV_LAC/EV_LAC/2438308o_pp_e2dsff_tcorr_AB_EV_LAC_EV_LAC_lbl.fits') #### PATH TO CHANGE ####
wave_start, wave_end = p1[1].data['WAVE_START'], p1[1].data['WAVE_END']
w1 = (p1[1].data['WAVE_START']+p1[1].data['WAVE_END'])/2.
depth = p1[1].data['LINE_DEPTH']

Prot = 4.3715
U = 0.05*Prot # Default is 0.05*Prot
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
ALL_dv = ALL_dv[:, used_waves]
ALL_sdv = ALL_sdv[:, used_waves]
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

tbinn, d2vbinn, sd2vbinn, dvbinn, sdvbinn = [], [], [], [], []
for idx in tqdm(range(ALL_d2v.shape[1])):
    ttemp, d2vtemp, sd2vtemp  = night_bin(times, ALL_d2v[:,idx], ALL_sd2v[:,idx])
    ttemp, dvtemp, sdvtemp  = night_bin(times, ALL_dv[:,idx], ALL_sdv[:,idx])
    d2vbinn.append(d2vtemp)
    sd2vbinn.append(sd2vtemp)
    dvbinn.append(dvtemp)
    sdvbinn.append(sdvtemp)
tbinn, d2vbinn, sd2vbinn = np.array(ttemp), np.array(d2vbinn).T, np.array(sd2vbinn).T
dvbinn, sdvbinn = np.array(dvbinn).T, np.array(sdvbinn).T

__, bervbin, __ = night_bin(times, BERV)

## Pre-Remove outliers by sigmaclipping

print('Outliers removal...')
d2vbinn = sigma_clip(d2vbinn, sigma=3, axis=0, masked=False)
dvbinn = sigma_clip(dvbinn, sigma=3, axis=0, masked=False)

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

## BERV -> Vtot

def odd_ratio_mean(value, err, odd_ratio = 1e-4, nmax = 10):
    #
    # Provide values and corresponding errors and compute a
    # weighted mean
    #
    #
    # odd_bad -> probability that the point is bad
    #
    # nmax -> number of iterations
    keep = np.isfinite(value)*np.isfinite(err)
    if np.sum(keep) == 0:
        return np.nan,np.nan

    value = value[keep]
    err = err[keep]
    guess = np.nanmedian(value)
    nite = 0
    while (nite < nmax):
        nsig = (value-guess)/err
        gg = np.exp(-0.5*nsig**2)
        odd_bad = odd_ratio/(gg+odd_ratio)
        odd_good = 1-odd_bad
        w = odd_good/err**2
        guess = np.nansum(value*w)/np.nansum(w)
        nite+=1

    bulk_error = np.sqrt(1/np.nansum(odd_good/err**2))
    return guess,bulk_error

RV = []
dRV = []
for i in tqdm(range(dvbinn.shape[0])):
    RV.append(odd_ratio_mean(dvbinn[i, :], sdvbinn[i, :])[0])
    dRV.append(odd_ratio_mean(dvbinn[i, :], sdvbinn[i, :])[1])
RV = np.array(RV)
dRV = np.array(dRV)

mean_rv = np.nanmean(RV)
Vtot = (mean_rv/1000.) - bervbin

## First PCA no correction


weights = 1. / dRV2
weights[np.isnan(RV2)] = 0
# Run pca
pca = WPCA(n_components=RV2.shape[1])
pca.regularization = 2
pca.fit(RV2, weights=weights)

fig, ax = plt.subplots(2, 3, figsize=(20, 5))
ax[0, 0].plot(Vtot, pca.components_[0], '.')
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
ax[0, 1].axvline(Prot, linestyle=':', color='b', alpha=0.5)
ax[0, 1].set_title('No correction')

ax[0, 2].plot(tbinn, pca.components_[0], '.')
ax[0, 2].set_xlabel('BJD')
ax[0, 2].set_ylabel('W1')

ax[1, 0].plot(Vtot, pca.components_[1], '.')
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
ax[1, 1].axvline(Prot, linestyle=':', color='b', alpha=0.5)

ax[1, 2].plot(tbinn, pca.components_[1], '.')
ax[1, 2].set_xlabel('BJD')
ax[1, 2].set_ylabel('W2')
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.5.0/0.7.275/TablesEV_LAC/BERV_Before.png')
plt.show()


# ## Imshow
#
# # plt.plot(Vtot, RV2.T, '.')
# # plt.show()
# plt.figure(figsize=(16,9))
# plt.imshow(RV2.T, origin='lower')
# plt.tight_layout()
# plt.show()

## Def function

def displine(ind):
    plt.figure(1)
    plt.plot(Vtot, RV2[ind].T, 'k.', alpha=0.2)
    plt.show()

def periodo(ind):
    plt.figure(2)
    t, dlw, sdlw = popnan(ind)
    frequency, power = LombScargle(t, dlw, sdlw).autopower(minimum_frequency=0.0005, maximum_frequency=1/1.5)
    plt.plot(1/frequency, power, 'r')
    plt.ylabel("power")
    plt.xscale('log')
    ls = LombScargle(t, dlw, sdlw)
    fap = ls.false_alarm_level(0.1)
    plt.axhline(fap, linestyle='-', color='k')
    fap = ls.false_alarm_level(0.01)
    plt.axhline(fap, linestyle='--', color='k')
    fap = ls.false_alarm_level(0.001)
    plt.axhline(fap, linestyle=':', color='k')
    plt.axvline(Prot, linestyle=':', color='g', alpha=0.5)
    plt.axvline(365.25, linestyle=':', color='r', alpha=0.5)
    plt.show()

def popnan(ind):
    RVout = RV2[ind, np.invert(np.isnan(RV2[ind]))]
    tout = tbinn[np.invert(np.isnan(RV2[ind]))]
    dRVout = dRV2[ind, np.invert(np.isnan(RV2[ind]))]
    return(tout, RVout, dRVout)

def maxperpow(ind):
    t, dlw, sdlw = popnan(ind)
    frequency, power = LombScargle(t, dlw, sdlw).autopower(minimum_frequency=0.0005, maximum_frequency=1/1.5)
    return(1/frequency[np.argmax(power)])

## Histogram of peak periodicities

powlist = [maxperpow(i) for i in tqdm(range(len(RV2)))]


# histogram on linear scale
plt.subplot(211)
hist, bins, _ = plt.hist(powlist, bins=1000)

# histogram on log scale.
# Use non-equal bin sizes, such that they look equal on log scale.
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.subplot(212)
plt.hist(powlist, bins=logbins)
plt.xscale('log')
plt.xlim(1, 1000)
plt.xlabel('P (d)')
plt.ylabel('#')
plt.axvline(Prot, c='r', alpha=0.2)
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.5.0/0.7.275/TablesEV_LAC/perpowhist.png')
plt.show()


## Select lines that peaks at Prot

indlw = []
for i in range(len(powlist)):
    if powlist[i]>Prot-U and powlist[i]<Prot+U:
        indlw.append(i)

RV2_filtr = RV2[indlw]
dRV2_filtr = dRV2[indlw]

## Run PCA after filtering

# wPCA
# weighting
weights = 1. / dRV2_filtr
weights[np.isnan(RV2_filtr)] = 0

# Run pca
pca = WPCA(n_components=RV2_filtr.shape[1])
pca.regularization = 2
pca.fit(RV2_filtr, weights=weights)

# Check Orthogonalization
print([np.dot(pca.components_[0], pca.components_[i]) for i in range(10)])

#Plot
fig, ax = plt.subplots(2, 3, figsize=(20, 5))
ax[0, 0].plot(bervbin, pca.components_[0], '.')
ax[0, 0].set_xlabel('BERV (m/s)')
ax[0, 0].set_ylabel('W1')

frequency, power = LombScargle(tbinn, pca.components_[0]).autopower(minimum_frequency=0.0005, maximum_frequency=1/1.5)#nyquist_factor=15)
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
ax[0, 1].axvline(Prot, linestyle=':', color='b', alpha=0.5)
ax[0, 1].set_title('LS periodogram filtered')

ax[0, 2].plot(tbinn, pca.components_[0], '.')
ax[0, 2].set_xlabel('BJD')
ax[0, 2].set_ylabel('W1')

ax[1, 0].plot(bervbin, pca.components_[1], '.')
ax[1, 0].set_xlabel('BERV (m/s)')
ax[1, 0].set_ylabel('W2')

frequency, power = LombScargle(tbinn, pca.components_[1]).autopower(minimum_frequency=0.0005, maximum_frequency=1/1.5)#nyquist_factor=15)
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
ax[1, 1].axvline(Prot, linestyle=':', color='b', alpha=0.5)

ax[1, 2].plot(tbinn, pca.components_[1], '.')
ax[1, 2].set_xlabel('BJD')
ax[1, 2].set_ylabel('W2')
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.5.0/0.7.275/TablesEV_LAC/LSfiltered.png')
plt.show()

# ## Already found by MerwPCAn ?
#
# our_list = w_used[indlw]
#
# Merbot, Merup = np.loadtxt('/home/paul/Bureau/IRAP/MerwPCAn/out/0.7.275/TablesEV_LAC/EV_LAC_impacted_lines_z0_2.rdb', usecols=(1,2),skiprows=2, dtype=str, unpack=True)
# Merbot = np.array(Merbot.astype(float))
# Merup = np.array(Merup.astype(float))
#
# ismer = []
# M = len(Merup)
# for i in tqdm(our_list):
#     inmer = False
#     m = 0
#     while inmer==False and m<M:
#         if i>Merbot[m] and i<Merup[m]:
#             inmer = True
#         m +=1
#     ismer.append(inmer)




