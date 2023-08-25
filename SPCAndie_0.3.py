## Imports

import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from tqdm import tqdm
from wpca import PCA, WPCA, EMPCA
from astropy.timeseries import LombScargle
frequency = np.linspace(1/1000, 1/1.1, 20000) # periodogram frequency grid
from astropy.stats import sigma_clip
import random
from sklearn.manifold import TSNE
import umap.umap_ as umap
import trimap
import pacmap
import matplotlib
from scipy import stats

## Load Data

path = '/media/paul/One Touch2/SPIRou_Data/0.7.275/CN_LEO/CN_LEO' #### PATH TO CHANGE ####
os.chdir(path)
ALL_d2v = []
ALL_sd2v  = []
ALL_dv = []
ALL_sdv  = []
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
ALL_dv  = np.array(ALL_dv)
ALL_sdv = np.array(ALL_sdv)
times = np.array(times)
BERV = np.array(BERV)

p1 = fits.open('/media/paul/One Touch2/SPIRou_Data/0.7.275/CN_LEO/CN_LEO/2376176o_pp_e2dsff_tcorr_AB_GL406_GL406_lbl.fits') #### PATH TO CHANGE ####
wave_start, wave_end = p1[1].data['WAVE_START'], p1[1].data['WAVE_END']
w1 = (p1[1].data['WAVE_START']+p1[1].data['WAVE_END'])/2.
depth = p1[1].data['LINE_DEPTH']

Prot = 2.704

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

print('Nigt binning...')
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
dvbinn, sdvbinn = [], []
for idx in tqdm(range(ALL_d2v.shape[1])):
    ttemp, d2vtemp, sd2vtemp  = night_bin(times, ALL_d2v[:,idx], ALL_sd2v[:,idx])
    d2vbinn.append(d2vtemp)
    sd2vbinn.append(sd2vtemp)
    __, dvtemp, sdvtemp  = night_bin(times, ALL_dv[:,idx], ALL_sdv[:,idx])
    dvbinn.append(dvtemp)
    sdvbinn.append(sdvtemp)

tbinn, d2vbinn, sd2vbinn = np.array(ttemp), np.array(d2vbinn).T, np.array(sd2vbinn).T
dvbinn, sdvbinn = np.array(dvbinn).T, np.array(sdvbinn).T

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
pca.fit(RV2, weights=weights)

fig, ax = plt.subplots(2, 3, figsize=(20, 5))
ax[0, 0].plot(bervbin, pca.components_[0], '.')
ax[0, 0].set_xlabel('BERV (km/s)')
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
ax[1, 0].set_xlabel('BERV (km/s)')
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
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/CN_Leo/BERV_Before.png')
plt.show()

## Def period function

def periodo(ind, show = False):
    t, dlw, sdlw = popnan(ind)
    ls = LombScargle(t, dlw, sdlw)#.autopower(minimum_frequency=0.0005, maximum_frequency=1/1.5)
    power = ls.power(frequency)
    max_freq = frequency[np.argmax(power)]
    return(power, 1/max_freq)

def popnan(ind):
    RVout = RV2[ind, np.invert(np.isnan(RV2[ind]))]
    tout = tbinn[np.invert(np.isnan(RV2[ind]))]
    dRVout = dRV2[ind, np.invert(np.isnan(RV2[ind]))]
    return(tout, RVout, dRVout)

## periodify

print('periodograms...')
# to para
powers = []
mf = []
for i in tqdm(range(RV2.shape[0])):
    powers.append(periodo(i)[0])
    mf.append(periodo(i)[1])
powers = np.array(powers)
mf = np.array(mf)

## Clustering Functions

# tSNE
def tSNEDisp(X, perplexity = 20, early_ex=12, learning_rate = 'auto', ):
    random.seed(110)
    X = X.reshape(X.shape[0], -1)
    X_embedded = TSNE(n_components=2, perplexity=perplexity, early_exaggeration=early_ex, learning_rate = learning_rate, random_state = 110).fit_transform(X)
    fig = plt.figure()
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], cmap="nipy_spectral", c=mf, s=2, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/CN_Leo/tSNE.png')
    plt.title('tSNE')
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
    fig = plt.figure()
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="nipy_spectral", c=mf, s=2, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.title('PacMap')
    plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/CN_Leo/PacMap.png')
    plt.show()
    return(X_transformed)

# Umap
def UmapDisp(X, n_neighbors=15, min_dist = 0.1):
    random.seed(110)
    X = X.reshape(X.shape[0], -1)
    X_embedded = umap.UMAP(n_neighbors=n_neighbors, min_dist = min_dist, n_components=2, random_state = 110).fit_transform(X)
    fig = plt.figure()
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1],cmap="nipy_spectral", c=mf, s=2, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.title('UMap')
    plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/CN_Leo/Umap.png')
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
    fig = plt.figure()
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="nipy_spectral", c=mf, s=2, norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.title('TriMap')
    plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/CN_Leo/TriMap.png')
    plt.show()
    return(X_transformed)

## Run
#
# print('tSNE...')
# tSNE_map = tSNEDisp(powers)
# print('Umap...')
# Umap_map = UmapDisp(powers)
print('TriMap...')
TriMap_map = TriMapDisp(powers)
print('PacMap...')
PacMap_map = PacMapDisp(powers)

## KDE sorting

def select_per(Per, mxfq):
    U = Per * 0.05
    indlw = []
    for i in range(len(mxfq)):
        if mxfq[i]>Per-U and mxfq[i]<Per+U:
            indlw.append(i)
    return(indlw)

an = 365.25

Pac_an  = PacMap_map[select_per(an, mf)]
Pac_an_2  = PacMap_map[select_per(an/2, mf)]
Pac_an_3  = PacMap_map[select_per(an/3, mf)]
Pac_an_4  = PacMap_map[select_per(an/4, mf)]
Pac_prot  = PacMap_map[select_per(Prot, mf)]
Pac_prot_2  = PacMap_map[select_per(Prot/2, mf)]
Pac_moon  = PacMap_map[select_per(28, mf)]
Pac_70 = PacMap_map[select_per(an/5, mf)]


plt.figure(10)
plt.plot(PacMap_map[:,0], PacMap_map[:,1], 'k.', alpha = 0.2)
plt.plot(Pac_an[:,0], Pac_an[:,1], '.', c='darkred', label='yr')
plt.plot(Pac_an_2[:,0], Pac_an_2[:,1], 'r.', label='yr/2')
plt.plot(Pac_an_3[:,0], Pac_an_3[:,1], '.', c='darkorange', label='yr/3')
plt.plot(Pac_an_4[:,0], Pac_an_4[:,1], '.', c='orange', label='yr/4')
plt.plot(Pac_70[:,0], Pac_70[:,1], 'y.', label='yr/5')
plt.plot(Pac_moon[:,0], Pac_moon[:,1], '.', c='lime', label='moon')
plt.plot(Pac_prot[:,0], Pac_prot[:,1], 'b.', label='Prot')
plt.plot(Pac_prot_2[:,0], Pac_prot_2[:,1], '.', c='purple', label='Prot/2')
plt.plot([4, 10], [0, 7], 'r', linestyle="--") # eq = PacMap_map[p,0]*0.075 - PacMap_map[p,1] > - 5
#plt.plot([-5, -1], [1, 5], 'r', linestyle="--") # eq = PacMap_map[p,0]*0.2 - PacMap_map[p,1] > 10
plt.legend()
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/CN_Leo/PacMapsomeperiods.png')
plt.show()

##
Tri_an  = TriMap_map[select_per(an, mf)]
Tri_an_2  = TriMap_map[select_per(an/2, mf)]
Tri_an_3  = TriMap_map[select_per(an/3, mf)]
Tri_an_4  = TriMap_map[select_per(an/4, mf)]
Tri_prot  = TriMap_map[select_per(Prot, mf)]
Tri_prot_2  = TriMap_map[select_per(Prot/2, mf)]
Tri_moon  = TriMap_map[select_per(28, mf)]
Tri_70 = TriMap_map[select_per(an/5, mf)]


plt.figure(13)
plt.plot(TriMap_map[:,0], TriMap_map[:,1], 'k.', alpha = 0.2)
plt.plot(Tri_an[:,0], Tri_an[:,1], '.', c='darkred', label='yr')
plt.plot(Tri_an_2[:,0], Tri_an_2[:,1], 'r.', label='yr/2')
plt.plot(Tri_an_3[:,0], Tri_an_3[:,1], '.', c='darkorange', label='yr/3')
plt.plot(Tri_an_4[:,0], Tri_an_4[:,1], '.', c='orange', label='yr/4')
plt.plot(Tri_70[:,0], Tri_70[:,1], 'y.', label='yr/5')
plt.plot(Tri_moon[:,0], Tri_moon[:,1], '.', c='lime', label='moon')
plt.plot(Tri_prot[:,0], Tri_prot[:,1], 'b.', label='Prot')
plt.plot(Tri_prot_2[:,0], Tri_prot_2[:,1], '.', c='purple', label='Prot/2')
plt.plot([120, -52], [20, -4], 'r', linestyle="--") # eq = PacMap_map[p,0]*0.075 - PacMap_map[p,1] > - 5
plt.legend()
plt.show()

## Filtering
# ## Paterns in TriMap

def isinellipse(x, y, h, k, a, b):
    rx2 = (x-h)**2 / a**2
    ry2 = (y-k)**2 / b**2
    return(rx2+ry2 <= 1)

plt.figure(14)
plt.plot(PacMap_map[:,0], PacMap_map[:,1], 'k.', alpha = 0.2)
# plt.axhline(-0.932, c='r')
#plt.axvline(35, c='r')
#plt.axvline(80, c='r')
#sns.kdeplot(PacMap_map[:,0], PacMap_map[:,1], cmap='hsv', levels=14)
plt.plot([4, 10], [0, 7], 'r', linestyle="--") # eq = PacMap_map[p,0]*0.075 - PacMap_map[p,1] > - 5
#plt.plot([-5, 5.5], [-6, 7], 'r', linestyle="--") # eq = PacMap_map[p,0]*0.075 - PacMap_map[p,1] > - 5
# plt.plot([-5, 9], [-6, 5.5], 'r', linestyle="--") # eq = PacMap_map[p,0]*0.15 - PacMap_map[p,1] > 5.3
# # plt.plot([-50, 15], [0, 0], 'r', linestyle="--")
#plt.axhline(-2.75, c='r')
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
    #isin = isinellipse(PacMap_map[p,0], PacMap_map[p,1], 2e-6, -2e-6, 1e-6, 0.2e-6)
    if  PacMap_map[p,0]* (7/6) - 4-2/3 > PacMap_map[p,1]  or PacMap_map[p,1] < -1 :#isinellipse(PacMap_map[p,0], PacMap_map[p,1], -4.25, -3.55, 1.35, 2) :
        isin = True
    #elif PacMap_map[p,0]* (1/3) < PacMap_map[p,1] + 1 and PacMap_map[p,0] > 5:# or PacMap_map[p,0]*slope > PacMap_map[p,1] - intercept :#isinellipse(PacMap_map[p,0], PacMap_map[p,1], -4.25, -3.55, 1.35, 2) :
     #   isin = True
    # elif PacMap_map[p,1] < -2.1e-6 and PacMap_map[p,0] > 2.5e-6:
    #     isin = True

    isincircle.append(isin)
    # else:
    #     isincircle.append(False)

#isincircle = np.invert(isincircle)
RV2_filtr = RV2[isincircle]
dRV2_filtr = dRV2[isincircle]
RV_filtr = dvbinn.T[isincircle]
dRV_filtr = sdvbinn.T[isincircle]
w_filtr = w_used[isincircle]

plt.plot(PacMap_map[isincircle,0], PacMap_map[isincircle,1], 'g.', alpha=0.5)
# plt.plot(PacMap_map[Bool_Merwaves,0], PacMap_map[Bool_Merwaves,1], 'r.', label='MerwPCAn z>3')
# plt.plot(PacMap_map[Bool_Bcorr,0], PacMap_map[Bool_Bcorr,1], 'y.', label='Bcorr R>0.3')
# plt.plot(PacMap_map[Bool_linescargle,0], PacMap_map[Bool_linescargle,1], 'b.', label='PCAeriodogram')
# plt.xlim(-8.5, 18)
# plt.ylim(-7.5, 8.5)
plt.show()

## PCA after filtering

weights = 1. / dRV2_filtr
weights[np.isnan(RV2_filtr)] = 0
# Run pca
pca = WPCA(n_components=RV2_filtr.shape[1])
pca.regularization = 2
pca.fit(RV2_filtr, weights=weights)

fig, ax = plt.subplots(2, 3, figsize=(20, 5))
ax[0, 0].plot(bervbin, pca.components_[0], '.')
ax[0, 0].set_xlabel('BERV (km/s)')
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

ax[1, 0].plot(bervbin, pca.components_[1], '.')
ax[1, 0].set_xlabel('BERV (km/s)')
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
plt.savefig('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/CN_Leo/PacMap_filtred.png')
plt.show()

## Saves

np.save('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/CN_Leo/readyforwPCA_d2vsd2v.npy', [RV2_filtr, dRV2_filtr]) #### PATH TO CHANGE ####
np.save('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/CN_Leo/readyforwPCA_dvsdv.npy', [RV_filtr, dRV_filtr]) #### PATH TO CHANGE ####
np.save('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/CN_Leo/readyforwPCA_linelist.npy', w_filtr)     #### PATH TO CHANGE ####
np.save('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/CN_Leo/readyforwPCA_epoc.npy', tbinn)          #### PATH TO CHANGE ####
np.save('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/CN_Leo/readyforwPCA_BERV.npy', bervbin)        #### PATH TO CHANGE ####

np.save('/home/paul/Bureau/IRAP/dLWPCA/SPCAndie/CN_Leo/firstcomponent.npy', pca.components_[:10])  #### PATH TO CHANGE ####