## Imports

import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.stats import norm
import random
from astropy.io import fits
from tqdm import tqdm
from wpca import PCA, WPCA, EMPCA
from scipy.stats import median_abs_deviation
from astropy.timeseries import LombScargle

## Load data
path = '/media/paul/One Touch2/SPIRou_Data/AU_MIC/AUMIC_AUMIC' #### PATH TO CHANGE ####
os.chdir(path)

file_list = []
ALL_d2v = []
ALL_sd2v = []
times = []
#dirs=directories
print('loading data...')
for (root, dirs, file) in os.walk(path):
    for f in tqdm(sorted(file)):
        if 'lbl.fits' in f:
            file_list.append(f)
            nthfile = fits.open(f)
            times.append(nthfile[0].header['BJD'])
            ALL_d2v.append(nthfile[1].data['d2v'])
            ALL_sd2v.append(nthfile[1].data['sd2v'])
            nthfile.close()

ALL_d2v = np.array(ALL_d2v)
ALL_sd2v = np.array(ALL_sd2v)
times = np.array(times)

p1 = fits.open('/media/paul/One Touch2/SPIRou_Data/AU_MIC/AUMIC_AUMIC/2812126o_pp_e2dsff_tcorr_AB_AUMIC_AUMIC_lbl.fits') #### PATH TO CHANGE ####

w1 = (p1[1].data['WAVE_START']+p1[1].data['WAVE_END'])/2.

file_list = [int(file_list[i][0:7]) for i in range(len(file_list))]

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
            drv_temp = [drv[index]]

    return res_times[:res_index], res_rv[:res_index], res_drv[:res_index]


tbinn, d2vbinn, sd2vbinn = [], [], []
for idx in tqdm(range(ALL_d2v.shape[1])):
    ttemp, d2vtemp, sd2vtemp  = night_bin(times, ALL_d2v[:,idx], ALL_sd2v[:,idx])
    d2vbinn.append(d2vtemp)
    sd2vbinn.append(sd2vtemp)
tbinn, d2vbinn, sd2vbinn = np.array(ttemp), np.array(d2vbinn).T, np.array(sd2vbinn).T

## Remove Outliers

print('Outliers removal...')
plt.figure(0)
stdbinn = []
for t in range(len(tbinn)):
    stdbinn.append(np.nanstd(d2vbinn[t]))
plt.plot(tbinn, stdbinn, 'ko')
plt.axhline(y=1e7, c='r')
plt.show()

def remove_outliers(T, D2V, sD2V, threshold):
    i = 0
    tuse = []
    while i < len(T):
        if np.nanstd(D2V[i]) > threshold:
            tuse.append(False)
        else:
            tuse.append(True)
        i += 1
    return(tuse)

tused = remove_outliers(tbinn, d2vbinn, sd2vbinn, 1e7)

tbinn, d2vbinn, sd2vbinn = tbinn[tused], d2vbinn[tused], sd2vbinn[tused]

## Normalization

print('Normalization...')
#mask NaNs for the averaging
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
#

## Saves

np.save('/home/paul/Bureau/IRAP/dLWPCA/out/TablesAU_MIC/readyforwPCA_d2vsd2v.npy', [RV2, dRV2]) #### PATH TO CHANGE ####
np.save('/home/paul/Bureau/IRAP/dLWPCA/out/TablesAU_MIC/readyforwPCA_linelist.npy', w_used)     #### PATH TO CHANGE ####
np.save('/home/paul/Bureau/IRAP/dLWPCA/out/TablesAU_MIC/readyforwPCA_epoc.npy', tbinn)          #### PATH TO CHANGE ####

## wPCA

print('runing PCA...')
# weighting

weights = 1. / dRV2
weights[np.isnan(RV2)] = 0

# Run pca

pca = WPCA(n_components=RV2.shape[1])

pca.regularization = 2

pca.fit(RV2, weights=weights)

# Check Orthogonalization

print([np.dot(pca.components_[0], pca.components_[i]) for i in range(10)])

# Periodograms of 3 first componant

fig, ax = plt.subplots(2, 2, figsize=(16, 6))
ax[0, 0].plot(tbinn, pca.components_[0], '.r', label='1st')
ax[0, 0].plot(tbinn, pca.components_[1], '.g', label='2nd')

frequency, power = LombScargle(tbinn, pca.components_[0]).autopower()
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

frequency, power = LombScargle(tbinn, pca.components_[1]).autopower()
ax[1, 0].plot(1/frequency, power, 'g')
ax[1, 0].set_xlabel("period (d)")
ax[1, 0].set_ylabel("power")
ax[1, 0].set_xscale('log')
ls = LombScargle(tbinn, pca.components_[1])
fap = ls.false_alarm_level(0.1)
ax[1, 0].axhline(fap, linestyle='-', color='k')
fap = ls.false_alarm_level(0.01)
ax[1, 0].axhline(fap, linestyle='--', color='k')
fap = ls.false_alarm_level(0.001)
ax[1, 0].axhline(fap, linestyle=':', color='k')

# Plot variance ratio
ax[1, 1].plot(np.arange(1, 11), pca.explained_variance_ratio_[:10])
ax[1, 1].set_xlim(1, 10)

ax[0, 0].set_title('2 first component')
ax[0, 1].set_title('1st Principal Vector')
ax[1, 0].set_title('2nd Principal Vector')
ax[1, 1].set_title('PCA variance ratio')
plt.show()

# save principal vectors

np.save('/home/paul/Bureau/IRAP/dLWPCA/out/TablesAU_MIC/2firstcomponent.npy', pca.components_[:2])  #### PATH TO CHANGE ####
