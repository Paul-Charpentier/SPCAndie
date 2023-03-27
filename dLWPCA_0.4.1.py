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
from hampel import hampel
import pandas as pd
from scipy.stats.stats import pearsonr
from PyAstronomy.pyasl import binningx0dt
import sys

## Load data
path = '/media/paul/One Touch2/SPIRou_Data/0.7.254/Gl_1289/Gl_1289' #### PATH TO CHANGE ####
os.chdir(path)
ALL_d2v = []
ALL_sd2v = []
times = []
BERV = []
#dirs=directories
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

p1 = fits.open('/media/paul/One Touch2/SPIRou_Data/0.7.254/Gl_1289/Gl_1289/2437300o_pp_e2dsff_tcorr_AB_GJ1289_GJ1289_lbl.fits') #### PATH TO CHANGE ####

w1 = (p1[1].data['WAVE_START']+p1[1].data['WAVE_END'])/2.

t_rdb, d2v_rdb, dd2v_rdb = np.loadtxt('/media/paul/One Touch2/SPIRou_Data/0.7.254/Gl_1289/lbl2_GJ1289_GJ1289.rdb', usecols=(0,3,4),skiprows=2, dtype=str, unpack=True)

t_rdb = np.array(t_rdb.astype(float)) + 2.4e6
d2v_rdb = np.array(d2v_rdb.astype(float))
dd2v_rdb = np.array(dd2v_rdb.astype(float))


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


## Pre-Remove outliers, Per times

print('Outliers removal...')
plt.figure(0)
stdbinn = []
for t in range(len(tbinn)):
    stdbinn.append(np.nanstd(d2vbinn[t]))
plt.plot(tbinn, stdbinn, 'ko')
m, s = np.mean(stdbinn), np.std(stdbinn)
plt.axhline(y=m+3*s, c='r')
plt.xlabel('BJD')
plt.ylabel('std')
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

tused = remove_outliers(tbinn, d2vbinn, sd2vbinn, m+3*s)

tbinn, d2vbinn, sd2vbinn, bervbin = tbinn[tused], d2vbinn[tused], sd2vbinn[tused], bervbin[tused]

## comparison w\ rdb

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

W, dW = [],[]
for t in tqdm(range(len(tbinn))):
    wt, dwt = odd_ratio_mean(d2vbinn[t], sd2vbinn[t])
    W.append(wt)
    dW.append(dwt)

W, dW = np.array(W), np.array(dW)


plt.figure(figsize=(16, 6))
ax1 = plt.subplot(311)
plt.errorbar(tbinn, W, yerr=dW, fmt='k.', label='me :)')
plt.errorbar(t_rdb, d2v_rdb, yerr=dd2v_rdb, fmt='b.', label = 'rdb')
plt.ylabel('$km^2 s^{-2}$')
plt.legend()
plt.tick_params('x', labelbottom=False)

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



t_com, Wcom, t_com, Wrdb, dWcom, dWrdb = reshapetocorr(tbinn, W, t_rdb, d2v_rdb, dd1=dW, dd2 = dd2v_rdb)
res = Wcom - Wrdb
dres = (dWcom+dWrdb)*0.5

ax2 = plt.subplot(312, sharex=ax1)
plt.errorbar(t_com, res, yerr=dres, fmt='b.')
plt.xlabel('RJD')
plt.ylabel('residuals')

ax3 = plt.subplot(313)
frequency, power = LombScargle(tbinn, W, dW).autopower()
plt.plot(1/frequency, power, 'k')
frequency, power = LombScargle(t_rdb, d2v_rdb, dd2v_rdb).autopower()
plt.plot(1/frequency, power, 'b')
plt.xlabel("period (d)")
plt.ylabel("power")
plt.xscale('log')

ls = LombScargle(tbinn, W, dW)
fap = ls.false_alarm_level(0.1)
plt.axhline(fap, linestyle='-', color='k')
fap = ls.false_alarm_level(0.01)
plt.axhline(fap, linestyle='--', color='k')
fap = ls.false_alarm_level(0.001)
plt.axhline(fap, linestyle=':', color='k')
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/rdbcomparison.png')  #### PATH TO CHANGE ####
plt.show()

print(pearsonr(Wcom, Wrdb))

## Berv correlation r\ Wave-length

def nanTremoval(X):
    used_waves = []
    for idx in range(len(X)):
        if np.isnan(X[idx])==0:
            used_waves.append(True)
        else:
            used_waves.append(False)

    return used_waves


BERV_CORR = []
for l in tqdm(range(d2vbinn.shape[1])):
    X = d2vbinn[:,l]
    BERV_CORR.append(pearsonr(X[nanTremoval(X)], bervbin[nanTremoval(X)])[0])
BERV_CORR = np.array(BERV_CORR)

binn = binningx0dt(w_used, BERV_CORR, nbins=50)[0]

stdbervall = np.std(BERV_CORR)

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True, gridspec_kw={'width_ratios': [7, 1]})

axs[0].plot(w_used, BERV_CORR, '.k', alpha=0.2)
axs[0].axhline(y=stdbervall, c='r')
axs[0].axhline(y=-stdbervall, c='r')
axs[0].errorbar(binn[:,0],binn[:,1],binn[:,2], fmt='ob')
axs[0].set_xlabel('$\lambda$ (nm)')
axs[0].set_ylabel('Correlation with BERV')
axs[0].set_ylim(-1,1)
axs[0].set_title('On the whole domain')

axs[1].hist(BERV_CORR, bins=20, orientation='horizontal')
axs[1].set_xlabel('#')
axs[1].set_ylim(-1,1)
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/BERVwl.png')  #### PATH TO CHANGE ####
plt.show()

# #### BREAK POINT
# sys.exit()
# #### ##### #####

## Same as previously but only where BERV < -24 m/s

def timebervselection(bandmin, bandmax, berv, time):
    tused = []
    for t in range(len(time)):
        if berv[t]< bandmax and berv[t]>bandmin:
            tused.append(True)
        else:
            tused.append(False)
    return(np.array(tused))

tselect = timebervselection(-32, -20, bervbin, tbinn)
RV2_bervselect, dRV2_bervselect, time_bervselect, berv_select = d2vbinn[tselect], sd2vbinn[tselect], tbinn[tselect], bervbin[tselect]

BERV_CORR = []
for l in tqdm(range(RV2_bervselect.shape[1])):
    X = RV2_bervselect[:,l]
    if np.sum(np.isnan(X))>=len(X)-1:
        BERV_CORR.append(np.nan)
    else :
        BERV_CORR.append(pearsonr(X[nanTremoval(X)], berv_select[nanTremoval(X)])[0])
BERV_CORR = np.array(BERV_CORR)

binn = binningx0dt(w_used, BERV_CORR, nbins=100)[0]

stdberv = np.nanstd(BERV_CORR)

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True, gridspec_kw={'width_ratios': [7, 1]})

axs[0].plot(w_used, BERV_CORR, '.k', alpha=0.2)
axs[0].axhline(y=stdberv, c='r')
axs[0].axhline(y=-stdberv, c='r')
axs[0].axhline(y=stdbervall, c='y', alpha=0.4)
axs[0].axhline(y=-stdbervall, c='y', alpha=0.4)
axs[0].errorbar(binn[:,0],binn[:,1],binn[:,2], fmt='ob')
axs[0].set_xlabel('$\lambda$ (nm)')
axs[0].set_ylabel('Correlation with BERV')
axs[0].set_ylim(-1,1)
axs[0].set_title('On the main BERV correlation')

axs[1].hist(BERV_CORR, bins=20, orientation='horizontal')
axs[1].set_xlabel('#')
axs[1].set_ylim(-1,1)
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/BERV_selected_wl.png')  #### PATH TO CHANGE ####
plt.show()

# #### BREAK POINT
# sys.exit()
# #### ##### #####

## Filter wavelength on theyr berv correlation ...

def hidewlberv(All_dw, Berv, threshold=0.5):
    usew = []
    for l in tqdm(range(d2vbinn.shape[1])):
        X = d2vbinn[:,l]
        if np.sum(np.isnan(X))>=len(X)-1:
            usew.append(False)
        else :
            BERVCORR = pearsonr(X[nanTremoval(X)], bervbin[nanTremoval(X)])
            if np.abs(BERVCORR[0]) > threshold:
                usew.append(False)
            else :
                usew.append(True)
    return usew


## On the BERV structure

usew = hidewlberv(RV2_bervselect, berv_select, threshold = stdbervall)
d2vberf = d2vbinn[:, usew]
sd2vberf = sd2vbinn[:, usew]
w_used = w_used[usew]
# ##
# BERV_CORR = []
# for l in tqdm(range(d2vberf.shape[1])):
#     X = d2vberf[:,l]
#     BERV_CORR.append(pearsonr(X[nanTremoval(X)], bervbin[nanTremoval(X)])[0])
# BERV_CORR = np.array(BERV_CORR)
# binn = binningx0dt(w_used, BERV_CORR, nbins=50)[0]
# #
# fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True, gridspec_kw={'width_ratios': [7, 1]})
#
# axs[0].plot(w_used, BERV_CORR, '.k', alpha=0.2)
# axs[0].axhline(y=stdbervall, c='r')
# axs[0].axhline(y=-stdbervall, c='r')
# axs[0].errorbar(binn[:,0],binn[:,1],binn[:,2], fmt='ob')
# axs[0].set_xlabel('$\lambda$ (nm)')
# axs[0].set_ylabel('Correlation with BERV')
# axs[0].set_ylim(-1,1)
# axs[0].set_title('On the whole domain')
#
# axs[1].hist(BERV_CORR, bins=20, orientation='horizontal')
# axs[1].set_xlabel('#')
# axs[1].set_ylim(-1,1)
# plt.show()

#
## Normalization

print('Normalization...')
#mask NaNs for the averaging
ma_dv = np.ma.MaskedArray(d2vberf.T, mask=np.isnan(d2vberf.T))
ma_sdv = np.ma.MaskedArray(sd2vberf.T, mask=np.isnan(sd2vberf.T))
#Compute average
avg_dv = np.ma.average(ma_dv, weights=1/ma_sdv**2, axis=1)
std_dv = np.ma.average((ma_dv-avg_dv.reshape(-1,1))**2, weights=1/ma_sdv**2, axis=1)
#Reshape average
avg_dv = avg_dv.data.reshape(-1,1)
std_dv = np.sqrt(std_dv.data.reshape(-1,1))
#Normalize
RV2 = (np.copy(d2vberf.T) - avg_dv)/std_dv
dRV2 = np.copy(sd2vberf.T)/std_dv
#
print('Binned shape:', RV2.shape)

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


## Show Berv


fig, ax = plt.subplots(1, 3, figsize=(20, 5))
ax[0].plot(bervbin, pca.components_[0], '.')
ax[0].set_xlabel('BERV (m/s)')
ax[0].set_ylabel('W1')
#
frequency, power = LombScargle(tbinn, pca.components_[0]).autopower()#nyquist_factor=15)
ax[1].plot(1/frequency, power, 'r')
ax[1].set_ylabel("power")
ax[1].set_xscale('log')
ls = LombScargle(tbinn, pca.components_[0])
fap = ls.false_alarm_level(0.1)
ax[1].axhline(fap, linestyle='-', color='k')
fap = ls.false_alarm_level(0.01)
ax[1].axhline(fap, linestyle='--', color='k')
fap = ls.false_alarm_level(0.001)
ax[1].axhline(fap, linestyle=':', color='k')
ax[1].axvline(74, linestyle=':', color='b', alpha=0.5)
ax[1].set_title('BERV filtered')

ax[2].plot(tbinn, pca.components_[0], '.')
ax[2].set_xlabel('BJD')
ax[2].set_ylabel('W1')
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/BERVwlfilter.png')  #### PATH TO CHANGE ####
plt.show()

#### BREAK POINT
sys.exit()
#### ##### #####


## MAD outliers wapiti removal
#
print('wapiting... ')
#
fig, ax = plt.subplots(2, 2, figsize=(16, 6))
ax[0, 0].plot(tbinn, pca.components_[0], 'r.', label='outliers')

ax[1, 0].plot(tbinn, pca.components_[1], 'r.', label='outliers')
#
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
#
frequency, power = LombScargle(tbinn, pca.components_[1]).autopower()
ax[1, 1].plot(1/frequency, power, 'r')
ax[1, 1].set_ylabel("power")
ax[1, 1].set_xscale('log')
ls = LombScargle(tbinn, pca.components_[0])
fap = ls.false_alarm_level(0.1)
ax[1, 1].axhline(fap, linestyle='-', color='k')
fap = ls.false_alarm_level(0.01)
ax[1, 1].axhline(fap, linestyle='--', color='k')
fap = ls.false_alarm_level(0.001)
ax[1, 1].axhline(fap, linestyle=':', color='k')



outlier_ind1 = hampel(pd.Series(np.copy(pca.components_[0])), window_size=10, n=5) #
outlier_ind2 = hampel(pd.Series(np.copy(pca.components_[1])), window_size=10, n=5)
tused = []
for t in range(len(tbinn)):
    if t in outlier_ind1 or t in outlier_ind2:
        tused.append(False)
    else :
        tused.append(True)
RV2, dRV2, tbinn, bervbin = RV2.T[tused], dRV2.T[tused], tbinn[tused], bervbin[tused]
RV2, dRV2 = RV2.T, dRV2.T
pcacomp = pca.components_.T[tused]
pcacomp = pcacomp.T

ax[0, 0].plot(tbinn, pcacomp[0], 'k.')
#
ax[1, 0].plot(tbinn, pcacomp[1], 'k.')

frequency, power = LombScargle(tbinn, pcacomp[0]).autopower()
ax[0, 1].plot(1/frequency, power, 'k')

frequency, power = LombScargle(tbinn, pcacomp[1]).autopower()
ax[1, 1].plot(1/frequency, power, 'k')

ax[0, 0].set_ylabel("W1")
ax[0, 0].set_xlabel('BJD')
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/MADoutlierafterBERV.png')    #### PATH TO CHANGE ####
plt.show()

#
#### BREAK POINT
sys.exit()
#### ##### #####

##wPCA
print('runing PCA...')
# weighting
#
weights = 1. / dRV2
weights[np.isnan(RV2)] = 0
#
# Run pca
#
pca = WPCA(n_components=RV2.shape[1])
#
pca.regularization = 2
#
pca.fit(RV2, weights=weights)
#
# Check Orthogonalization

print([np.dot(pca.components_[0], pca.components_[i]) for i in range(10)])

pcacomp = pca.components_
# # Check berv
#
plt.figure(11)
plt.title('After BERV Correction')
plt.plot(bervbin, pca.components_[0], '.')
plt.xlabel('BERV (m/s)')
plt.ylabel('W1')
#plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289AfterBERVcorrection.png') #### PATH TO CHANGE ####
plt.show()
print('BERV corrected shape:', RV2.shape)





## Saves

print('final shape:', RV2.shape)
np.save('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/readyforwPCA_d2vsd2v.npy', [RV2, dRV2]) #### PATH TO CHANGE ####
np.save('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/readyforwPCA_linelist.npy', w_used)     #### PATH TO CHANGE ####
np.save('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/readyforwPCA_epoc.npy', tbinn)          #### PATH TO CHANGE ####
np.save('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/readyforwPCA_BERV.npy', bervbin)        #### PATH TO CHANGE ####


fig, ax = plt.subplots(2, 2, figsize=(16, 6))
ax[0, 0].plot(tbinn, pcacomp[0], '.r', label='1st')
ax[0, 0].plot(tbinn, pcacomp[1], '.g', label='2nd')

frequency, power = LombScargle(tbinn, pcacomp[0]).autopower()#nyquist_factor=25)
ax[0, 1].plot(1/frequency, power, 'r')
ax[0, 1].set_ylabel("power")
ax[0, 1].set_xscale('log')
ls = LombScargle(tbinn, pcacomp[0])
fap = ls.false_alarm_level(0.1)
ax[0, 1].axhline(fap, linestyle='-', color='k')
fap = ls.false_alarm_level(0.01)
ax[0, 1].axhline(fap, linestyle='--', color='k')
fap = ls.false_alarm_level(0.001)
ax[0, 1].axhline(fap, linestyle=':', color='k')
ax[0, 1].axvline(74, linestyle=':', color='b', alpha=0.5)
#
frequency, power = LombScargle(tbinn, pcacomp[1]).autopower()#nyquist_factor=25)
ax[1, 0].plot(1/frequency, power, 'g')
ax[1, 0].set_xlabel("period (d)")
ax[1, 0].set_ylabel("power")
ax[1, 0].set_xscale('log')
ls = LombScargle(tbinn, pcacomp[1])
fap = ls.false_alarm_level(0.1)
ax[1, 0].axhline(fap, linestyle='-', color='k')
fap = ls.false_alarm_level(0.01)
ax[1, 0].axhline(fap, linestyle='--', color='k')
fap = ls.false_alarm_level(0.001)
ax[1, 0].axhline(fap, linestyle=':', color='k')
ax[1, 0].axvline(74, linestyle=':', color='b', alpha=0.5)
#
# Plot variance ratio
ax[1, 1].plot(np.arange(1, 11), pca.explained_variance_ratio_[:10])
ax[1, 1].set_xlim(1, 10)
#
ax[0, 0].set_title('2 first component')
ax[0, 1].set_title('1st Principal Vector')
ax[1, 0].set_title('2nd Principal Vector')
ax[1, 1].set_title('PCA variance ratio')
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/FinalPCA.png')        #### PATH TO CHANGE ####
plt.show()
#

np.save('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/2firstcomponent.npy', pcacomp[:10])  #### PATH TO CHANGE ####