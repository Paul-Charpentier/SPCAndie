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

## Load data

path = '/media/paul/One Touch2/SPIRou_Data/0.7.254/Gl_1289/Gl_1289' #### PATH TO CHANGE ####
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

## Pre-Remove outliers ([NOTE] Find a clever way)

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
#

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
ax[0, 1].axvline(74, linestyle=':', color='b', alpha=0.5)
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
ax[1, 1].axvline(74, linestyle=':', color='b', alpha=0.5)

ax[1, 2].plot(tbinn, pca.components_[1], '.')
ax[1, 2].set_xlabel('BJD')
ax[1, 2].set_ylabel('W2')

plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/BERV_Before.png')  #### PATH TO CHANGE ####
plt.show()

## Total dLW as a function of wavelength

print('Filter by mean BERV drift...')

dwl = []
stdwl = []
for l in tqdm(range(d2vbinn.shape[1])):
    X = d2vbinn[:,l]
    dwl.append(np.nanmean(X))
    stdwl.append(np.nanstd(X))
dwl = np.array(dwl)

binn = binningx0dt(w_used, dwl, nbins=50)[0]

stdbervall = np.std(dwl)

fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True, gridspec_kw={'width_ratios': [7, 1]})

axs[0].plot(w_used, dwl, '.k', alpha=0.2)
axs[0].axhline(y=stdbervall, c='r')
axs[0].axhline(y=-stdbervall, c='r')
axs[0].errorbar(binn[:,0],binn[:,1],binn[:,2], fmt='ob')
axs[0].set_xlabel('$\lambda$ (nm)')
axs[0].set_ylabel('<dLW>')
axs[0].set_title('On the whole domain')
axs[1].set_ylim(-10*stdbervall, 10*stdbervall)

axs[1].hist(dwl, bins=200, orientation='horizontal')
axs[1].set_xlabel('#')
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/meandLW.png')  #### PATH TO CHANGE ####
plt.show()

## Same as previously but on BERV Slices

#Berv domain
BERV_Domain = np.linspace(np.min(bervbin)-1, np.max(bervbin)+1, 9)

def timebervselection(bandmin, bandmax, berv, time):
    tused = []
    for t in range(len(time)):
        if berv[t]< bandmax and berv[t]>bandmin:
            tused.append(True)
        else:
            tused.append(False)
    return(np.array(tused))

def meanslice(slice_begin, slice_end, d2v, sd2v, t,  berv):
    tselect = timebervselection(slice_begin, slice_end, berv, t)
    RV2_bervselect, dRV2_bervselect, time_bervselect, berv_select = d2vbinn[tselect], sd2vbinn[tselect], tbinn[tselect], bervbin[tselect]
    mslc = []
    for l in range(RV2_bervselect.shape[1]):
        X = RV2_bervselect[:,l]
        if np.sum(np.isnan(X))>=len(X)-1:
            mslc.append(np.nan)
        else :
            mslc.append(np.nanmean(X))
    mslc = np.array(mslc)
    slicetd = np.nanstd(mslc)
    return(mslc, slicetd)

W_rmv = []
fig, axs = plt.subplots(3, len(BERV_Domain)-1, tight_layout=True, sharey='row')
axs[0, 0].set_ylabel('$<dLW>_{slice}$')
axs[1, 0].set_ylabel('$\Delta <dLW>_{slice}$')
axs[2, 0].set_ylabel('W1')
for i in tqdm(range(1, len(BERV_Domain))):
    h = i-1
    start = BERV_Domain[h]
    end = BERV_Domain[i]

    mslc, slctd = meanslice(start, end, d2vbinn, sd2vbinn, tbinn, bervbin)
    stddif = np.nanstd(mslc - dwl)
    W_rmv.append( np.abs(mslc - dwl) > stddif )

    axs[0, h].plot(w_used, mslc, '.k', alpha=0.2)
    axs[0, h].axhline(y=slctd, c='r', alpha=0.5)
    axs[0, h].axhline(y=-slctd, c='r', alpha=0.5)
    axs[0, h].axhline(y=0, c='r', ls=':', alpha=0.5)
    axs[0, h].set_xlabel('$\lambda$ (nm)')
    axs[0, h].set_ylim(-10*slctd, 10*slctd)

    axs[1, h].plot(w_used, mslc - dwl, '.b', alpha=0.2)
    axs[1, h].axhline(y=stddif, c='r', alpha=0.5)
    axs[1, h].axhline(y=-stddif, c='r', alpha=0.5)
    axs[1, h].axhline(y=0, c='r', ls=':', alpha=0.5)
    axs[1, h].set_xlabel('$\lambda$ (nm)')
    axs[1, h].set_ylim(-10*slctd, 10*slctd)

    axs[2, h].plot(bervbin, pca.components_[0], '.')
    axs[2, h].axvline(start, linestyle=':', color='g', alpha=0.5)
    axs[2, h].axvline(end, linestyle=':', color='g', alpha=0.5)
    axs[2, h].set_xlabel('BERV')

plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/BERV_mean_slice_wl.png')  #### PATH TO CHANGE ####
plt.show()

W_rmv = np.array(W_rmv)
sumrvm = [np.sum(W_rmv[i, :])*100/d2vbinn.shape[1] for i in range(len(W_rmv))]

## Filter

def filtrmean(d2vb, sd2vb, bervb, wbool, bervdom, tbinn=None):
    d2v, sd2v, berv = np.copy(d2vb), np.copy(sd2vb), np.copy(bervb)
    for t in tqdm(range(len(d2v))):
        i=0
        stopcond = 0
        while stopcond == 0:
            i += 1
            if berv[t] < bervdom[i]:
                for l in range(len(d2v[t])):
                    if wbool[i-1][l] == True:
                        d2v[t][l] = np.nan
                        sd2v[t][l] = np.nan
                stopcond = 1
            if i > 9:
                print('domain error')
                return(-1)
    return(d2v, sd2v)

d2vfiltr, sd2vfiltr = filtrmean(d2vbinn, sd2vbinn, bervbin, W_rmv, BERV_Domain)

# re-remove nans
used_waves = remove_nan(d2vfiltr, threshold = len(tbinn)//2 + len(tbinn)//10)
d2vfiltr = d2vfiltr[:, used_waves]
sd2vfiltr = sd2vfiltr[:, used_waves]
w_used = w_used[used_waves]

## Second PCA after mean drift filtering

print('Second PCA after mean drift filtering...')
# Normalization
#mask NaNs for the averaging
ma_dv = np.ma.MaskedArray(d2vfiltr.T, mask=np.isnan(d2vfiltr.T))
ma_sdv = np.ma.MaskedArray(sd2vfiltr.T, mask=np.isnan(sd2vfiltr.T))
#Compute average
avg_dv = np.ma.average(ma_dv, weights=1/ma_sdv**2, axis=1)
std_dv = np.ma.average((ma_dv-avg_dv.reshape(-1,1))**2, weights=1/ma_sdv**2, axis=1)
#Reshape average
avg_dv = avg_dv.data.reshape(-1,1)
std_dv = np.sqrt(std_dv.data.reshape(-1,1))
#Normalize
RV2 = (np.copy(d2vfiltr.T) - avg_dv)/std_dv
dRV2 = np.copy(sd2vfiltr.T)/std_dv

# wPCA
# weighting
weights = 1. / dRV2
weights[np.isnan(RV2)] = 0

# Run pca
pca = WPCA(n_components=RV2.shape[1])
pca.regularization = 2
pca.fit(RV2, weights=weights)

# Check Orthogonalization
print([np.dot(pca.components_[0], pca.components_[i]) for i in range(10)])

#Plot
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
ax[0, 1].axvline(74, linestyle=':', color='b', alpha=0.5)
ax[0, 1].set_title('BERV $\Delta <dLW>_{slice}$ filtered')

ax[0, 2].plot(tbinn, pca.components_[0], '.')
ax[0, 2].set_xlabel('BJD')
ax[0, 2].set_ylabel('W2')

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
ax[1, 1].axvline(74, linestyle=':', color='b', alpha=0.5)

ax[1, 2].plot(tbinn, pca.components_[1], '.')
ax[1, 2].set_xlabel('BJD')
ax[1, 2].set_ylabel('W2')

plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/Dmeanwlfilter.png')  #### PATH TO CHANGE ####
plt.show()

## Total BERV correlation x wavelength

def nanTremoval(X):
    used_waves = []
    for idx in range(len(X)):
        if np.isnan(X[idx])==0:
            used_waves.append(True)
        else:
            used_waves.append(False)
    return used_waves

BERV_CORR = []
for l in tqdm(range(d2vfiltr.shape[1])):
    X = d2vfiltr[:,l]
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

## Same as previously but on BERV Slices

def corrslice(slice_begin, slice_end, d2v, sd2v, t,  berv):
    tselect = timebervselection(slice_begin, slice_end, berv, t)
    RV2_bervselect, dRV2_bervselect, time_bervselect, berv_select = d2v[tselect], sd2v[tselect], t[tselect], berv[tselect]
    BERV_CORR = []
    for l in range(RV2_bervselect.shape[1]):
        X = RV2_bervselect[:,l]
        if np.sum(np.isnan(X))>=len(X)-1:
            BERV_CORR.append(np.nan)
        else :
            BERV_CORR.append(pearsonr(X[nanTremoval(X)], berv_select[nanTremoval(X)])[0])
    BERV_CORR = np.array(BERV_CORR)
    stdberv = np.nanstd(BERV_CORR)
    return(BERV_CORR, stdberv)

W_rmv = []
fig, axs = plt.subplots(2, len(BERV_Domain)-1, tight_layout=True)
axs[0, 0].set_ylabel('Correlation with BERV')
axs[1, 0].set_ylabel('W1')
for i in tqdm(range(1, len(BERV_Domain))):
    h = i-1
    start = BERV_Domain[h]
    end = BERV_Domain[i]
    BERV_CORR, stdberv = corrslice(start, end, d2vfiltr, sd2vfiltr, tbinn, bervbin)
    W_rmv.append( np.abs(BERV_CORR) > stdberv )

    axs[0, h].plot(w_used, BERV_CORR, '.k', alpha=0.2)
    axs[0, h].axhline(y=stdberv, c='r')
    axs[0, h].axhline(y=-stdberv, c='r')
    axs[0, h].axhline(y=stdbervall, c='y', alpha=0.4)
    axs[0, h].axhline(y=-stdbervall, c='y', alpha=0.4)
    axs[0, h].set_xlabel('$\lambda$ (nm)')
    axs[0, h].set_ylim(-1,1)

    axs[1, h].plot(bervbin, pca.components_[0], '.')
    axs[1, h].axvline(start, linestyle=':', color='g', alpha=0.5)
    axs[1, h].axvline(end, linestyle=':', color='g', alpha=0.5)
    axs[1, h].set_xlabel('BERV')

plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/BERV_slice_wl.png')  #### PATH TO CHANGE ####
plt.show()

W_rmv = np.array(W_rmv)
sumrvm = [np.sum(W_rmv[i, :])*100/d2vfiltr.shape[1] for i in range(len(W_rmv))]

## Filter after it

def filtrcorr(d2vb, sd2vb, bervb, wbool, bervdom, tbinn=None):
    d2v, sd2v, berv = np.copy(d2vb), np.copy(sd2vb), np.copy(bervb)
    for t in tqdm(range(len(d2v))):
        i=0
        stopcond = 0
        while stopcond == 0:
            i += 1
            if berv[t] < bervdom[i]:
                for l in range(len(d2v[t])):
                    if wbool[i-1][l] == True:
                        d2v[t][l] = np.nan
                        sd2v[t][l] = np.nan
                stopcond = 1
            if i > 9:
                print('domain error')
                return(-1)
    return(d2v, sd2v)

D2Vfiltr, sD2Vfiltr = filtrcorr(d2vfiltr, sd2vfiltr, bervbin, W_rmv, BERV_Domain)

# re-remove nans
used_waves = remove_nan(D2Vfiltr, threshold = len(tbinn)//2 + len(tbinn)//3)
D2Vfiltr = D2Vfiltr[:, used_waves]
sD2Vfiltr = sD2Vfiltr[:, used_waves]
w_used = w_used[used_waves]

## Third PCA after BERV Correlation filtering

print('Third PCA after BERV Correlation filtering...')
#mask NaNs for the averaging
ma_dv = np.ma.MaskedArray(D2Vfiltr.T, mask=np.isnan(D2Vfiltr.T))
ma_sdv = np.ma.MaskedArray(sD2Vfiltr.T, mask=np.isnan(sD2Vfiltr.T))
#Compute average
avg_dv = np.ma.average(ma_dv, weights=1/ma_sdv**2, axis=1)
std_dv = np.ma.average((ma_dv-avg_dv.reshape(-1,1))**2, weights=1/ma_sdv**2, axis=1)
#Reshape average
avg_dv = avg_dv.data.reshape(-1,1)
std_dv = np.sqrt(std_dv.data.reshape(-1,1))
#Normalize
RV2 = (np.copy(D2Vfiltr.T) - avg_dv)/std_dv
dRV2 = np.copy(sD2Vfiltr.T)/std_dv

# wPCA
# weighting
weights = 1. / dRV2
weights[np.isnan(RV2)] = 0

# Run pca
pca = WPCA(n_components=RV2.shape[1])
pca.regularization = 2
pca.fit(RV2, weights=weights)

# Check Orthogonalization
print([np.dot(pca.components_[0], pca.components_[i]) for i in range(10)])

#Plot
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
ax[0, 1].axvline(74, linestyle=':', color='b', alpha=0.5)
ax[0, 1].set_title('+ BERV Correlation filtered')

ax[0, 2].plot(tbinn, pca.components_[0], '.')
ax[0, 2].set_xlabel('BJD')
ax[0, 2].set_ylabel('W2')

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
ax[1, 1].axvline(74, linestyle=':', color='b', alpha=0.5)

ax[1, 2].plot(tbinn, pca.components_[1], '.')
ax[1, 2].set_xlabel('BJD')
ax[1, 2].set_ylabel('W2')

plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/BERVwlfilter.png')  #### PATH TO CHANGE ####
plt.show()

## MAD outliers wapitICH removal

print('wapiting... ')

fig, ax = plt.subplots(2, 2, figsize=(16, 6))
ax[0, 0].plot(tbinn, pca.components_[0], 'r.', label='outliers')
ax[1, 0].plot(tbinn, pca.components_[1], 'r.', label='outliers')

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
D2Vfiltr, sD2Vfiltr = D2Vfiltr[tused], sD2Vfiltr[tused]
RV2, dRV2 = RV2.T, dRV2.T
pcacomp = pca.components_.T[tused]
pcacomp = pcacomp.T

ax[0, 0].plot(tbinn, pcacomp[0], 'k.')
ax[1, 0].plot(tbinn, pcacomp[1], 'k.')

frequency, power = LombScargle(tbinn, pcacomp[0]).autopower()
ax[0, 1].plot(1/frequency, power, 'k')
ax[0, 1].axvline(74, linestyle=':', color='b', alpha=0.5)

frequency, power = LombScargle(tbinn, pcacomp[1]).autopower()
ax[1, 1].plot(1/frequency, power, 'k')
ax[1, 1].axvline(74, linestyle=':', color='b', alpha=0.5)
ax[0, 0].set_ylabel("W1")
ax[0, 0].set_xlabel('BJD')
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/MADoutlierafterBERV.png')    #### PATH TO CHANGE ####
plt.show()

## Last PCA after wapitiching

print('Last PCA after wapitiching')

# weighting
weights = 1. / dRV2
weights[np.isnan(RV2)] = 0

# Run pca
pca = WPCA(n_components=RV2.shape[1])
pca.regularization = 2
pca.fit(RV2, weights=weights)

# Check Orthogonalization
print([np.dot(pca.components_[0], pca.components_[i]) for i in range(10)])

# Plot
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
ax[0, 1].axvline(74, linestyle=':', color='b', alpha=0.5)
ax[0, 1].set_title('(+ MAD Correction) Final')

ax[0, 2].plot(tbinn, pca.components_[0], '.')
ax[0, 2].set_xlabel('BJD')
ax[0, 2].set_ylabel('W2')

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
ax[1, 1].axvline(74, linestyle=':', color='b', alpha=0.5)

ax[1, 2].plot(tbinn, pca.components_[1], '.')
ax[1, 2].set_xlabel('BJD')
ax[1, 2].set_ylabel('W2')

plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/out_0.4.0/0.7.254/TablesGL1289/BERVwlfinal.png')  #### PATH TO CHANGE ####
plt.show()