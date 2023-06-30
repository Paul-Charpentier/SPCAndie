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

## First PCA without any correction

print('First PCA without any correction...')

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

weights = 1. / dRV2
weights[np.isnan(RV2)] = 0

# Run pca
pca = WPCA(n_components=RV2.shape[1])
pca.regularization = 2
pca.fit(RV2, weights=weights)

fig, ax = plt.subplots(2, 3, figsize=(20, 5))
ax[0, 0].plot(bervbin, pca.components_[0], '.')
ax[0, 0].set_xlabel('BERV (m/s)')
ax[0, 0].set_ylabel('W1')

frequency, power = LombScargle(tbinn, pca.components_[0]).autopower() #nyquist_factor=15)
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
ax[0, 1].set_title('No correction')

ax[0, 2].plot(tbinn, pca.components_[0], '.')
ax[0, 2].set_xlabel('BJD')
ax[0, 2].set_ylabel('W2')

ax[1, 0].plot(bervbin, pca.components_[1], '.')
ax[1, 0].set_xlabel('BERV (m/s)')
ax[1, 0].set_ylabel('W2')

frequency, power = LombScargle(tbinn, pca.components_[1]).autopower() #nyquist_factor=15)
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

plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.275/TablesEV_LAC/BERV_Before.png')  #### PATH TO CHANGE ####
plt.show()

## Load B

times_B, B, dB = np.loadtxt('/media/paul/One Touch2/SPIRou_Data/0.7.275/EV_LAC/bf_time_ev_lac.txt',
                   usecols=(1,15,16),skiprows=1, dtype=str, unpack=True)

times_B = np.array(times_B.astype(float))
B = np.array(B.astype(float))
dB = np.array(dB.astype(float))

frequency, power = LombScargle(times_B, B, dB).autopower(nyquist_factor=15)
plt.plot(1/frequency, power, 'r')
plt.ylabel("power")
plt.xscale('log')
ls = LombScargle(tbinn, pca.components_[0])
fap = ls.false_alarm_level(0.1)
plt.axhline(fap, linestyle='-', color='k')
fap = ls.false_alarm_level(0.01)
plt.axhline(fap, linestyle='--', color='k')
fap = ls.false_alarm_level(0.001)
plt.axhline(fap, linestyle=':', color='k')
plt.axvline(4.3715, linestyle=':', color='b', alpha=0.5)
plt.title('<B>')
plt.show()
## Same grid for corelation

def reshapetocorr(t1, d1, t2, d2, dd1 = None, dd2 = None, verbose = False):

    i, j = 0, 0
    D1 = np.copy(d1)
    T1 = np.copy(t1)
    D2 = np.copy(d2)
    T2 = np.copy(t2)
    if dd1 is not None:
        dD1 = np.copy(dd1)
    if dd2 is not None:
        dD2 = np.copy(dd2)
    while i < len(T1) and j < len(T2):
        diff = T1[i] - T2[j]
        if np.abs(diff) < 0.5:
            i += 1
            j += 1
        elif diff > 0.5:
            if verbose :
                print('L', T1[i], 'R', T2[j])
                print('point remove R')
            T2 = np.delete(T2, j)
            D2 = np.delete(D2, j)
            if dd2 is not None:
                dD2 = np.delete(dD2, j)
        elif diff < -0.5:
            if verbose :
                print('L', T1[i], 'R', T2[j])
                print('point remove L')
            T1 = np.delete(T1, i)
            D1 = np.delete(D1, i)
            if dd1 is not None:
                dD1 = np.delete(dD1, j)

    T1 = T1[:i]
    D1 = D1[:i]
    T2 = T2[:j]
    D2 = D2[:j]
    if dd1 is not None:
        dD1 = dD1[:j]
    if dd2 is not None:
        dD2 = dD2[:j]
    if dd1 is None:
        dD1 = None
    if dd2 is None:
        dD2 = None

    return(T1, D1, T2, D2, dD1, dD2)

def popnan(ind):
    RVout = RV2[ind, np.invert(np.isnan(RV2[ind]))]
    tout = tbinn[np.invert(np.isnan(RV2[ind]))]
    dRVout = dRV2[ind, np.invert(np.isnan(RV2[ind]))]
    return(tout, RVout, dRVout)


## Run pearsonR

corr = []

for l in tqdm(range(RV2.shape[0])):
    tnonan, dwnonan, ddwnonan = popnan(l)
    twcorr, wcorr, tbcorr, bcorr, dwcorr, dbcorr = reshapetocorr(tnonan, dwnonan, times_B, B, dd1 = ddwnonan, dd2 = dB)
    corr.append(pearsonr(bcorr, wcorr)[0])

fig, ax = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [5, 1]})
ax[0].plot(w_used, corr, 'k.')
ax[0].axhline(0.3, c='r')
#hist, bins, __ = plt.hist(corr, bins=100)
#ax[1].plot(hist, bins)
ax[1].hist(corr, bins=69, orientation='horizontal')
plt.show()

## select lines that correlates

filtered_waves = [(corr[l] > 0.3) for l in range(len(corr))]

w_filtr = w_used[filtered_waves]
d2v_filtr = RV2[filtered_waves]
sd2v_filtr = dRV2[filtered_waves]

## Run second PCA after filtering

print('Second PCA with correction...')

weights = 1. / sd2v_filtr
weights[np.isnan(d2v_filtr)] = 0

# Run pca
pca = WPCA(n_components=d2v_filtr.shape[1])
pca.regularization = 2
pca.fit(d2v_filtr, weights=weights)

fig, ax = plt.subplots(2, 3, figsize=(20, 5))
ax[0, 0].plot(bervbin, pca.components_[0], '.')
ax[0, 0].set_xlabel('BERV (m/s)')
ax[0, 0].set_ylabel('W1')

frequency, power = LombScargle(tbinn, pca.components_[0]).autopower()#nyquist_factor=15)
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
ax[0, 1].set_title('<B> cor filtered')

ax[0, 2].plot(tbinn, pca.components_[0], '.')
ax[0, 2].set_xlabel('BJD')
ax[0, 2].set_ylabel('W1')

ax[1, 0].plot(bervbin, pca.components_[1], '.')
ax[1, 0].set_xlabel('BERV (m/s)')
ax[1, 0].set_ylabel('W2')

frequency, power = LombScargle(tbinn, pca.components_[1]).autopower()#nyquist_factor=15)
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

plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.275/TablesEV_LAC/B_after.png')  #### PATH TO CHANGE ####
plt.show()

##

def line_score_selection(z0, display=True):
    filtered_waves = [(corr[l] > z0) for l in range(len(corr))]

    w_filtr = w_used[filtered_waves]
    d2v_filtr = RV2[filtered_waves]
    sd2v_filtr = dRV2[filtered_waves]

    weights = 1. / sd2v_filtr
    weights[np.isnan(d2v_filtr)] = 0

    # Fit WPCA model
    pca = WPCA(n_components=d2v_filtr.shape[1])
    pca.fit(d2v_filtr, weights=weights)

    #Plot

    frequency, power = LombScargle(tbinn, pca.components_[0]).autopower(nyquist_factor=10)
    ls = LombScargle(tbinn, pca.components_[0])
    if display:
        plt.clf()
        plt.plot(1/frequency, power, 'k-')
    fap =ls.false_alarm_probability(power.max())
    # false alarm levels
    fap = ls.false_alarm_level(1e-3)
    if display:
        plt.axhline(y=fap,linestyle="-",color="k")
    fap = ls.false_alarm_level(1e-5)
    if display:
        plt.axhline(y=fap,linestyle="--",color="k")
        plt.xscale('log')
        plt.annotate('C0 = ' + str(z0), (110, 0.225))
        plt.annotate('#  = ' + str(d2v_filtr.shape[0]), (110, 0.25))
        plt.ylim(0, 0.55)
    return(w_filtr, 1/frequency[np.argmax(power)], np.max(power))

## Gif
#
fig = plt.figure(figsize=(16, 9))
WOAWANIMTROBELLE = animation.FuncAnimation(fig, line_score_selection, frames = np.arange(0.8, 0, -0.005))#np.arange(len(times_tcorr)))
WOAWANIMTROBELLE.save('/home/paul/Bureau/IRAP/dLWPCA/out_0.5.0/0.7.275/TablesEV_LAC/cscore.gif', writer=animation.PillowWriter(fps=24))
plt.show()

## Log fap evolution

LOGFAP = []
MAXFRE = []
NBRLIN = []
for z in tqdm(np.arange(0.8, 0, -0.005)):
    impacted, maxfreq, logfap = line_score_selection(z, display=False)
    MAXFRE.append(maxfreq)
    LOGFAP.append(logfap)
    NBRLIN.append(len(impacted))


fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
ax[0].plot(0.8-np.arange(0.8, 0, -0.005), LOGFAP)
ax[0].set_ylabel('max power')

ax[1].plot(0.8-np.arange(0.8, 0, -0.005), MAXFRE)
ax[1].set_ylabel('frequency')
ax[1].set_yscale('log')

ax[2].plot(0.8-np.arange(0.8, 0, -0.005), NBRLIN)
ax[2].set_ylabel('# lines')
ax[2].set_xlabel('0.8-c0')
ax[2].set_yscale('log')
plt.show()


## Save

np.save('/home/paul/Bureau/IRAP/dLWPCA/out_0.5.0/0.7.275/TablesEV_LAC/Bcorr_d2vsd2v.npy', [d2v_filtr, sd2v_filtr]) #### PATH TO CHANGE ####
np.save('/home/paul/Bureau/IRAP/dLWPCA/out_0.5.0/0.7.275/TablesEV_LAC/Bcorr_linelist.npy', w_filtr)     #### PATH TO CHANGE ####
np.save('/home/paul/Bureau/IRAP/dLWPCA/out_0.5.0/0.7.275/TablesEV_LAC/Bcorr_epoc.npy', tbinn)          #### PATH TO CHANGE ####
#np.save('/home/paul/Bureau/IRAP/dLWPCA/out_0.5.0/0.7.275/TablesEV_LAC/Bcorr_indlinlist.npy', indlw)        #### PATH TO CHANGE ####
np.save('/home/paul/Bureau/IRAP/dLWPCA/out_0.5.0/0.7.275/TablesEV_LAC/Bcorr_firstcomponent.npy', pca.components_[:10])  #### PATH TO CHANGE ####

## Load filling factors

f0, f2, f4, f6, f8, f10, ef0, ef2, ef4, ef6, ef8, ef10 = np.loadtxt('/media/paul/One Touch2/SPIRou_Data/0.7.275/EV_LAC/bf_time_ev_lac.txt',
                   usecols=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),skiprows=1, dtype=str, unpack=True)


f0 = np.array(f0.astype(float))
f2 = np.array(f2.astype(float))
f4 = np.array(f4.astype(float))
f6 = np.array(f6.astype(float))
f8 = np.array(f8.astype(float))
f10 = np.array(f10.astype(float))
ef0 = np.array(ef0.astype(float))
ef2 = np.array(ef2.astype(float))
ef4 = np.array(ef4.astype(float))
ef6 = np.array(ef6.astype(float))
ef8 = np.array(ef8.astype(float))
ef10 = np.array(ef10.astype(float))

plt.figure(10)
plt.errorbar(times_B, f0, ef0, fmt='.', label='f0')
plt.errorbar(times_B, f2, ef2, fmt='.', label='f2')
plt.errorbar(times_B, f4, ef4, fmt='.', label='f4')
plt.errorbar(times_B, f6, ef6, fmt='.', label='f6')
plt.errorbar(times_B, f8, ef8, fmt='.', label='f8')
plt.errorbar(times_B, f10, ef10, fmt='.', label='f10')
plt.legend()
plt.xlabel('BJD')
plt.show()

## Correlation with filling factors

#f0
twcorr, wcorr, tf0corr, f0corr, __, df0corr = reshapetocorr(tbinn, pca.components_[0], times_B, f0, dd2 = ef0)
print('f0', pearsonr(f0corr, wcorr))
#f2
twcorr, wcorr, tf2corr, f2corr, __, df2corr = reshapetocorr(tbinn, pca.components_[0], times_B, f2, dd2 = ef2)
print('f2', pearsonr(f2corr, wcorr))
#f4
twcorr, wcorr, tf4corr, f4corr, __, df4corr = reshapetocorr(tbinn, pca.components_[0], times_B, f4, dd2 = ef4)
print('f4', pearsonr(f4corr, wcorr))
#f6
twcorr, wcorr, tf6corr, f6corr, __, df6corr = reshapetocorr(tbinn, pca.components_[0], times_B, f6, dd2 = ef6)
print('f6', pearsonr(f6corr, wcorr))
#f8
twcorr, wcorr, tf8corr, f8corr, __, df8corr = reshapetocorr(tbinn, pca.components_[0], times_B, f8, dd2 = ef8)
print('f8', pearsonr(f8corr, wcorr))
#f10
twcorr, wcorr, tf10corr, f10corr, __, df10corr = reshapetocorr(tbinn, pca.components_[0], times_B, f10, dd2 = ef10)
print('f10', pearsonr(f10corr, wcorr))