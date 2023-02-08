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

RV2, dRV2 = np.load('/home/paul/Bureau/IRAP/TablesAU_MIC/readyforwPCA_d2vsd2v.npy')
w_used = np.load('/home/paul/Bureau/IRAP/TablesAU_MIC/readyforwPCA_linelist.npy')
times = np.load('/home/paul/Bureau/IRAP/TablesAU_MIC/readyforwPCA_epoc.npy')

## Plot

for i in range(10):
    plt.errorbar(times, RV2.T[i], yerr=dRV2.T[i], fmt='.')
plt.show()

## wPCA

def plot_results(ThisPCA, X, weights=None, Xtrue=None, ncomp=3):
    # Compute the standard/weighted PCA
    if weights is None:
        kwds = {}
    else:
        kwds = {'weights': weights}

    # Compute the PCA vectors & variance
    pca = ThisPCA(n_components=10).fit(X, **kwds)

    # Reconstruct the data using the PCA model
    Y = ThisPCA(n_components=ncomp).fit_reconstruct(X, **kwds)

    # Create the plots
    fig, ax = plt.subplots(2, 2, figsize=(16, 6))
    if Xtrue is not None:
        ax[0, 0].plot(Xtrue[:10].T, c='gray', lw=1)
        ax[1, 1].plot(Xtrue[:10].T, c='gray', lw=1)
    ax[0, 0].plot(X[:10].T, c='black', lw=1)
    ax[1, 1].plot(Y[:10].T, c='black', lw=1)

    ax[0, 1].plot(pca.components_[:ncomp].T, c='black')

    ax[1, 0].plot(np.arange(1, 11), pca.explained_variance_ratio_)
    ax[1, 0].set_xlim(1, 10)
    ax[1, 0].set_ylim(0, None)

    ax[0, 0].xaxis.set_major_formatter(plt.NullFormatter())
    ax[0, 1].xaxis.set_major_formatter(plt.NullFormatter())

    ax[0, 0].set_title('Input Data')
    ax[0, 1].set_title('First {0} Principal Vectors'.format(ncomp))
    ax[1, 1].set_title('Reconstructed Data ({0} components)'.format(ncomp))
    ax[1, 0].set_title('PCA variance ratio')
    ax[1, 0].set_xlabel('principal vector')
    ax[1, 0].set_ylabel('proportion of total variance')

    fig.suptitle(ThisPCA.__name__, fontsize=16)
    plt.show()
    return(pca)

## weighting

weights = 1. / dRV2
weights[np.isnan(RV2)] = 0

## RUN !

eigen2vectors = plot_results(WPCA, RV2.T, weights=weights.T, ncomp=4)

## Save result

np.save('/home/paul/Bureau/IRAP/TablesAU_MIC/2firstcomponent.npy', eigen2vectors.components_[:4])
