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
import random
from sklearn.manifold import TSNE
import umap.umap_ as umap
import trimap
import pacmap
import seaborn as sns
frequency = np.linspace(1/1000, 1/1.1, 20000) # periodogram frequency grid
from scipy import stats

## Load Data

path = '/media/paul/One Touch2/SPIRou_Data/0.7.275/Gl_905/GL905' #### PATH TO CHANGE ####
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

p1 = fits.open('/media/paul/One Touch2/SPIRou_Data/0.7.275/Gl_905/GL905/2413152o_pp_e2dsff_tcorr_AB_GL905_GL905_lbl.fits') #### PATH TO CHANGE ####
wave_start, wave_end = p1[1].data['WAVE_START'], p1[1].data['WAVE_END']
w1 = (p1[1].data['WAVE_START']+p1[1].data['WAVE_END'])/2.
depth = p1[1].data['LINE_DEPTH']

Prot = 110

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
#
# ## Del ALL NaNs
#
# used_waves = remove_nan(d2vbinn, threshold = 0)
# d2vbinn = d2vbinn[:, used_waves]
# sd2vbinn = sd2vbinn[:, used_waves]
# w_used = w_used[used_waves]

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

## BERV -> Vtot

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

ax[1, 0].plot(Vtot, pca.components_[1], '.')
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
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/Gl905/BERV_Before.png')
plt.show()

## Def period function

def periodo(ind, show = False):
    plt.figure(2)
    t, dlw, sdlw = popnan(ind)
    ls = LombScargle(t, dlw, sdlw)#.autopower(minimum_frequency=0.0005, maximum_frequency=1/1.5)
    power = ls.power(frequency)
    if show == True:
        plt.plot(1/frequency, power, 'r')
        plt.ylabel("power")
        plt.xscale('log')
        fap = ls.false_alarm_level(0.1)
        plt.axhline(fap, linestyle='-', color='k')
        fap = ls.false_alarm_level(0.01)
        plt.axhline(fap, linestyle='--', color='k')
        fap = ls.false_alarm_level(0.001)
        plt.axhline(fap, linestyle=':', color='k')
        plt.axvline(Prot, linestyle=':', color='g', alpha=0.5)
        plt.axvline(365.25, linestyle=':', color='r', alpha=0.5)
        plt.show()
    return(power)

def popnan(ind):
    RVout = RV2[ind, np.invert(np.isnan(RV2[ind]))]
    tout = tbinn[np.invert(np.isnan(RV2[ind]))]
    dRVout = dRV2[ind, np.invert(np.isnan(RV2[ind]))]
    return(tout, RVout, dRVout)

## periodify
print('periodograms...')
powers = []
for i in tqdm(range(RV2.shape[0])):
    powers.append(periodo(i))

powers = np.array(powers)

## Clustering Functions

# tSNE

def tSNEDisp(X, perplexity = 20, early_ex=12, learning_rate = 'auto', ):
    random.seed(110)
    X = X.reshape(X.shape[0], -1)
    X_embedded = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_ex, learning_rate = learning_rate, random_state = 110).fit_transform(X)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], cmap="Spectral", c=w_used, s=2)
    plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/Gl905/tSNE.png')
    plt.show()
    return(X_embedded)

# PacMap

def PacMapDisp(X, n_neighbors=10, MN_rate=0.5, FP_rate=2.0):
    random.seed(110)
    X = X.reshape(X.shape[0], -1)
    # Initialize the pacmap instance
    # Setting n_neighbors to "None" leads to an automatic parameter selection
    # choice shown in "parameter" section of the README file.
    # Notice that from v0.6.0 on, we rename the n_dims parameter to n_components.
    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=n_neighbors, MN_ratio=MN_rate, FP_ratio=FP_rate, random_state = 110, apply_pca = True)
    # fit the data (The index of transformed data corresponds to the index of the original data)
    X_transformed = embedding.fit_transform(X, init="pca")
    # visualize the embedding
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="Spectral", c=w_used, s=2)
    plt.title('PacMap')
    plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/Gl905/PacMap.png')
    plt.show()
    return(X_transformed)

# Umap

def UmapDisp(X, n_neighbors=15, min_dist = 0.1):
    random.seed(110)
    X = X.reshape(X.shape[0], -1)
    X_embedded = umap.UMAP(n_neighbors=n_neighbors, min_dist = min_dist, n_components=2, random_state = 110).fit_transform(X)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X_embedded[:, 0], X_embedded[:, 1], cmap="Spectral", c=w_used, s=2)
    plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/Gl905/Umap.png')
    plt.show()
    return(X_embedded)

# TriMap

def TriMapDisp(X, n_in = 12, n_out=10, n_random = 3):
    random.seed(110)
    X = X.reshape(X.shape[0], -1)
    # Initialize the pacmap instance
    # Setting n_neighbors to "None" leads to an automatic parameter selection
    # choice shown in "parameter" section of the README file.
    # Notice that from v0.6.0 on, we rename the n_dims parameter to n_components.
    embedding = trimap.TRIMAP(n_dims=2, n_inliers=n_in, n_outliers=n_out, n_random=n_random)
    # fit the data (The index of transformed data corresponds to the index of the original data)
    X_transformed = embedding.fit_transform(X)
    # visualize the embedding
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="Spectral", c=w_used, s=2)
    plt.title('TriMap')
    plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/Gl905/TriMap.png')
    plt.show()
    return(X_transformed)

## Run

# print('tSNE...')
# tSNE_map = tSNEDisp(powers)
# print('Umap...')
# Umap_map = UmapDisp(powers)
print('TriMap...')
TriMap_map = TriMapDisp(powers)
print('PacMap...')
PacMap_map = PacMapDisp(powers)

## Paterns in TriMap

def isinellipse(x, y, h, k, a, b):
    rx2 = (x-h)**2 / a**2
    ry2 = (y-k)**2 / b**2
    return(rx2+ry2 <= 1)

plt.figure(10)
plt.plot(PacMap_map[:,0], PacMap_map[:,1], 'k.', alpha = 0.2)
# plt.axhline(-0.932, c='r')
#plt.axvline(35, c='r')
#plt.axvline(80, c='r')
sns.kdeplot(PacMap_map[:,0], PacMap_map[:,1], cmap='hsv', levels=14)
# plt.plot([-5, 0.9, 5], [2, -2.9, 6.75], 'r', linestyle="--") # eq = PacMap_map[p,0]*0.075 - PacMap_map[p,1] > - 5
# plt.plot([4, 10.75], [0, 6.75], 'r', linestyle="--") # eq = PacMap_map[p,0]*0.2 - PacMap_map[p,1] > 10
# plt.plot([-5, 9], [-6, 5.5], 'r', linestyle="--") # eq = PacMap_map[p,0]*0.15 - PacMap_map[p,1] > 5.3
# # plt.plot([-50, 15], [0, 0], 'r', linestyle="--")
#plt.axhline(4.5, c='r')
#plt.show()


# plt.axvline(2.5e-6, c='r')
#plt.axhline(-1.95e-6, c='r')
# plt.axhline(-1.9e-6, c='r')
# #
# theta = np.linspace( 0 , 2 * np.pi , 150 )
# radius = 0.3e-6
# a = radius*4 * np.cos( theta )
# b = radius * np.sin( theta )
# plt.plot(a + 1.9e-6,b - 2e-6,c='r')
slope, intercept, __, __, __  = stats.linregress(PacMap_map.T)

isincircle = []
for p in tqdm(range(PacMap_map.shape[0])):
    isin = False
    #isin = isinellipse(tSNE_map[p,0], tSNE_map[p,1], 2e-6, -2e-6, 1e-6, 0.2e-6)
    if isinellipse(PacMap_map[p,0], PacMap_map[p,1], -4.25, -3.55, 1.35, 2) :#PacMap_map[p,0]* (7.25/4) > PacMap_map[p,1] + 3 :#PacMap_map[p,0]*slope < PacMap_map[p,1] - intercept :#
        isin = True
    # elif tSNE_map[p,1] < -2.1e-6 and tSNE_map[p,0] > 2.5e-6:
    #     isin = True

    isincircle.append(isin)
    # else:
    #     isincircle.append(False)

#isincircle = np.invert(isincircle)
RV2_filtr = RV2[isincircle]
dRV2_filtr = dRV2[isincircle]
w_filtr = w_used[isincircle]

plt.plot(PacMap_map[isincircle,0], PacMap_map[isincircle,1], 'g.', alpha=0.5)
# plt.xlim(-8.5, 18)
# plt.ylim(-7.5, 8.5)
plt.show()

# ## PCA after filtering


weights = 1. / dRV2_filtr
weights[np.isnan(RV2_filtr)] = 0
# Run pca
pca = WPCA(n_components=RV2_filtr.shape[1])
pca.regularization = 2
pca.fit(RV2_filtr, weights=weights)

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

ax[0, 1].set_title('PacMap filtered')

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
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/Gl905/tSNE_filtred.png')
plt.show()

#### BREAK POINT
sys.exit()
#### ##### #####

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
RV2_filtr, dRV2_filtr, tbinn, bervbin, Vtot = RV2_filtr.T[tused], dRV2_filtr.T[tused], tbinn[tused], bervbin[tused], Vtot[tused]
# D2Vfiltr, sD2Vfiltr = D2Vfiltr[tused], sD2Vfiltr[tused]
# DVfiltr, sDVfiltr = DVfiltr[tused], sDVfiltr[tused]
RV2_filtr, dRV2_filtr = RV2_filtr.T, dRV2_filtr.T
pcacomp = pca.components_.T[tused]
pcacomp = pcacomp.T

ax[0, 0].plot(tbinn, pcacomp[0], 'k.')
ax[1, 0].plot(tbinn, pcacomp[1], 'k.')

frequency, power = LombScargle(tbinn, pcacomp[0]).autopower()
ax[0, 1].plot(1/frequency, power, 'k')
ax[0, 1].axvline(Prot, linestyle=':', color='b', alpha=0.5)

frequency, power = LombScargle(tbinn, pcacomp[1]).autopower()
ax[1, 1].plot(1/frequency, power, 'k')
ax[1, 1].axvline(Prot, linestyle=':', color='b', alpha=0.5)

ax[0, 0].set_ylabel("W1")
ax[0, 0].set_xlabel('BJD')
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/Gl905/MADoutlierafterBERV.png')    #### PATH TO CHANGE ####
plt.show()

# ## Last PCA after wapitiching

print('Last PCA after wapitiching')

weights = 1. / dRV2_filtr
weights[np.isnan(RV2_filtr)] = 0
# Run pca
pca = WPCA(n_components=RV2_filtr.shape[1])
pca.regularization = 2
pca.fit(RV2_filtr, weights=weights)

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
ax[0, 1].set_title('PacMap filtered')

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
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/Gl905/tSNE_filtred.png')
plt.show()
