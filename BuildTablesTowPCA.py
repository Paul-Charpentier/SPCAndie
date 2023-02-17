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

## Load data
path = '/media/paul/One Touch2/SPIRou_Data/AU_MIC/AUMIC_AUMIC'
os.chdir(path)

file_list = []
ALL_d2v = []
ALL_sd2v = []
times = []
#dirs=directories
# #print('loading ...')
for (root, dirs, file) in os.walk(path):
    for f in tqdm(file):
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

p1 = fits.open('/media/paul/One Touch2/SPIRou_Data/AU_MIC/AUMIC_AUMIC/2812126o_pp_e2dsff_tcorr_AB_AUMIC_AUMIC_lbl.fits')

w1 = (p1[1].data['WAVE_START']+p1[1].data['WAVE_END'])/2.

## Remove some NaNs

def remove_nan(X, threshold = 200):

    used_waves = []
    for idx in tqdm(range(X.shape[1])):
        if np.sum(np.isnan(X[:,idx]))<=threshold:
            used_waves.append(True)
        else:
            used_waves.append(False)

    return used_waves

used_waves = remove_nan(ALL_d2v)
ALL_d2v = ALL_d2v[:, used_waves]
ALL_sd2v = ALL_sd2v[:, used_waves]
w_used = w1[used_waves]

## Night Binn

intnuit = [np.round(t) for t in times]
i = 1
while i < len(intnuit):
    if intnuit[i] in intnuit[:i]:
        intnuit.pop(i)
    else :
        i += 1

tbinn = []
rvbinn = []

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

tbinn = []
d2vbinn = []
sd2vbinn = []

for rjd in tqdm(intnuit):
    samenight = (np.abs(rjd-times) < 0.5)
    tbinn.append(np.mean(times[samenight]))
    Y = []
    sY = []
    for l in range(ALL_d2v.shape[1]):
        y, sy = odd_ratio_mean(ALL_d2v[samenight, l], ALL_sd2v[samenight, l])

        Y.append(y)
        sY.append(sy)
    d2vbinn.append(Y)
    sd2vbinn.append(sY)

d2vbinn = np.array(d2vbinn)
sd2vbinn = np.array(sd2vbinn)

## Normalization

#mask NaNs for the averaging
ma_dv = np.ma.MaskedArray(d2vbinn, mask=np.isnan(d2vbinn))
ma_sdv = np.ma.MaskedArray(sd2vbinn, mask=np.isnan(sd2vbinn))
#Compute average
avg_dv = np.ma.average(ma_dv, weights=1/ma_sdv**2, axis=1)
std_dv = np.ma.average((ma_dv-avg_dv.reshape(-1,1))**2, weights=1/ma_sdv**2, axis=1)
#Reshape average
avg_dv = avg_dv.data.reshape(-1,1)
std_dv = np.sqrt(std_dv.data.reshape(-1,1))
#Normalize
RV2 = (np.copy(d2vbinn) - avg_dv)/std_dv
dRV2 = (np.copy(sd2vbinn))/std_dv


## Saves

np.save('/home/paul/Bureau/IRAP/TablesAU_MIC/readyforwPCA_d2vsd2v.npy', [RV2, dRV2])
np.save('/home/paul/Bureau/IRAP/TablesAU_MIC/readyforwPCA_linelist.npy', w_used)
np.save('/home/paul/Bureau/IRAP/TablesAU_MIC/readyforwPCA_epoc.npy', tbinn)



## Build Table without any NaNs

#Load Data
path = '/media/paul/One Touch2/SPIRou_Data/AU_MIC/AUMIC_AUMIC'
os.chdir(path)

file_list = []
ALL_d2v = []
ALL_sd2v = []
times = []
#dirs=directories
# #print('loading ...')
for (root, dirs, file) in os.walk(path):
    for f in tqdm(file):
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

p1 = fits.open('/media/paul/One Touch2/SPIRou_Data/AU_MIC/AUMIC_AUMIC/2812126o_pp_e2dsff_tcorr_AB_AUMIC_AUMIC_lbl.fits')

w1 = (p1[1].data['WAVE_START']+p1[1].data['WAVE_END'])/2.

# remove all nans
used_waves = remove_nan(ALL_d2v, threshold=0)
ALL_d2v = ALL_d2v[:, used_waves]
ALL_sd2v = ALL_sd2v[:, used_waves]
w_used = w1[used_waves]

# Night binn
intnuit = [np.round(t) for t in times]
i = 1
while i < len(intnuit):
    if intnuit[i] in intnuit[:i]:
        intnuit.pop(i)
    else :
        i += 1

tbinn = []
d2vbinn = []
sd2vbinn = []

for rjd in tqdm(intnuit):
    samenight = (np.abs(rjd-times) < 0.5)
    tbinn.append(np.mean(times[samenight]))
    Y = []
    sY = []
    for l in range(ALL_d2v.shape[1]):
        y, sy = odd_ratio_mean(ALL_d2v[samenight, l], ALL_sd2v[samenight, l])

        Y.append(y)
        sY.append(sy)
    d2vbinn.append(Y)
    sd2vbinn.append(sY)

d2vbinn = np.array(d2vbinn)
sd2vbinn = np.array(sd2vbinn)

# normalize
#mask NaNs for the averaging
ma_dv = np.ma.MaskedArray(d2vbinn, mask=np.isnan(d2vbinn))
ma_sdv = np.ma.MaskedArray(sd2vbinn, mask=np.isnan(sd2vbinn))
#Compute average
avg_dv = np.ma.average(ma_dv, weights=1/ma_sdv**2, axis=1)
std_dv = np.ma.average((ma_dv-avg_dv.reshape(-1,1))**2, weights=1/ma_sdv**2, axis=1)
#Reshape average
avg_dv = avg_dv.data.reshape(-1,1)
std_dv = std_dv.data.reshape(-1,1)
#Normalize
RV2 = (np.copy(d2vbinn) - avg_dv)/std_dv
dRV2 = (np.copy(sd2vbinn) - avg_dv)/std_dv


# save

np.save('/home/paul/Bureau/IRAP/TablesAU_MIC/readyforwPCA_d2vsd2v_NoNaN.npy', [RV2, dRV2])
np.save('/home/paul/Bureau/IRAP/TablesAU_MIC/readyforwPCA_linelist_NoNaN.npy', w_used)
np.save('/home/paul/Bureau/IRAP/TablesAU_MIC/readyforwPCA_epoc_NoNaN.npy', tbinn)
