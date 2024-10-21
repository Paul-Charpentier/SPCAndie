## Imports

import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from wpca import PCA, WPCA, EMPCA
from astropy.timeseries import LombScargle
import random
import pacmap
import matplotlib
from scipy import stats
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay
import sys
sys.path.append(os.path.abspath("/media/paul/One Touch11/wapiti_workflow"))
from wapiti import wapiti_tools, wapiti
from astropy.table import Table
from collections import Counter
from tqdm import tqdm
import gc

## Set Up
#  (this is the only part you are suppose to change unless you know what you do)

Target = ['EV_LAC', 'GL410']
Prot = [4.3615, 13.87]
Template = Target
#Niters = 750 #Number of iterations for PacMap (default is 450)

chatty = True
plotty = True
initial_mad_rejection = 10   # MAD rejection threshold for the outlier removal
n_component_rejection = 10  # Number of component to check for the outlier removal
looplier = 10 # Number of outlier rejection loops

yr = 365.25
period_grid = np.logspace(np.log(1.25), 3, 1000)
frequency = 1/period_grid # periodogram frequency grid

cwd = "/media/paul/One Touch11/wapiti_workflow" ## base work directory


## Functions

def get_data(target, template):
    #files = os.listdir(cwd+'/lblrv/'+target+'_'+template)
    os.chdir(cwd+'/lblrv/'+target+'_'+template)
    times_lbl = []
    d2vs_all = []
    dd2vs_all = []
    snrs = []
    berv = []
    for (root, dirs, files) in os.walk(cwd+'/lblrv/'+target+'_'+template):
        for f in tqdm(sorted(files)):
            if 'lbl.fits' in f:
                lbl = fits.open(f, memmap=False)
                # = fits.open(cwd+'/lblrv/'+target+'_'+template+'/'+file)
                times_lbl.append(lbl[0].header['BJD'])
                berv.append(lbl[0].header['BERV'])
                d2vs_all.append(lbl[1].data['d2v'])
                dd2vs_all.append(lbl[1].data['sd2v'])
                snrs.append(lbl[0].header['SPEMSNR'])
                lbl.close()
    times_lbl = np.array(times_lbl)
    d2vs_all = np.array(d2vs_all)
    dd2vs_all = np.array(dd2vs_all)
    snrs = np.array(snrs)
    berv = np.array(berv)
    return times_lbl, d2vs_all, dd2vs_all, snrs, berv

def run_PCA(x, dx):
    # Create masked arrays for the used RVs and RV uncertainties, masking any NaN values
    ma_x = np.ma.MaskedArray((x.T), mask=np.isnan((x.T)))
    ma_dx = np.ma.MaskedArray((dx.T), mask=np.isnan((dx.T)))

    # Compute the average and variance of the used RVs and RV uncertainties using the masked arrays
    average = np.ma.average(ma_x, weights=1/ma_dx**2, axis=1)
    variance = np.ma.average((ma_x-average.reshape(-1, 1))**2, weights=1/ma_dx**2, axis=1)

    # Reshape the averages and standard deviations into column vectors
    mean_x = average.data.reshape(-1, 1)
    std_x = np.sqrt(variance.data.reshape(-1, 1))

    # Normalize the used RVs and RV uncertainties
    X = (np.copy(x.T)-mean_x)/std_x
    dX = np.copy(dx.T)/std_x

    weights = 1. / dX # Not a mistake - The wPCA is coded in a way that 1/dX as a weights variable here means using 1/dX² as an effective weights
    weights[np.isnan(X)] = 0
    pca_x = WPCA(n_components=X.shape[1])
    pca_x.fit(X, weights=weights)
    return(pca_x)

def periodo(ind, X, dX, t, show = False):
    t, dlw, sdlw = popnan(ind, X, dX, t)
    ls = LombScargle(t, dlw, sdlw)#.autopower(minimum_frequency=0.0005, maximum_frequency=1/1.5)
    power = ls.power(frequency)
    max_freq = frequency[np.argmax(power)]
    max_pow = np.max(power)
    max_fap = -np.log(ls.false_alarm_probability(max_pow))
    return(power, 1/max_freq, max_pow, max_fap)

def popnan(ind, X, dX, t):
    Xout = X[ind, np.invert(np.isnan(X[ind]))]
    tout = t[np.invert(np.isnan(X[ind]))]
    dXout = dX[ind, np.invert(np.isnan(X[ind]))]
    return(tout, Xout, dXout)

def PacMapDisp(X, max_freq, max_pow, max_fap, target_name, niters = 450, n_neighbors=10, MN_rate=0.5, FP_rate=2.0):
    random.seed(110)
    X = X.reshape(X.shape[0], -1)
    # Initialize the pacmap instance
    # Setting n_neighbors to "None" leads to an automatic parameter selection
    # choice shown in "parameter" section of the README file.
    # Notice that from v0.6.0 on, we rename the n_dims parameter to n_components.
    embedding = pacmap.PaCMAP(num_iters = niters, n_components=2, n_neighbors=n_neighbors, MN_ratio=MN_rate, FP_ratio=FP_rate, random_state = 110, apply_pca = True)
    # fit the data (The index of transformed data corresponds to the index of the original data)
    X_transformed = embedding.fit_transform(X, init="pca")
    # visualize the embedding
    if plotty:
        fig = plt.figure(figsize=(16,10))
        scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], cmap="nipy_spectral", c=max_freq, s=10*max_fap, norm=matplotlib.colors.LogNorm())
        cb = plt.colorbar(label='Period [d]')
        cb.set_label(label='Period [d]', size='large', weight='bold')
        cb.ax.tick_params(labelsize='large')
        handles, labels = scatter.legend_elements(prop='sizes', alpha = 0.6)
        plt.legend(handles, labels, title = '$- 10 logFAP_{max}$', prop={'size':12, 'weight':'bold'})
        #plt.title('PacMap')
        plt.axis('off')
        plt.savefig(cwd+"/out/" + target_name + "PacMap.pdf", format="pdf", bbox_inches="tight")
        #plt.ion()
    return(X_transformed)

def run_pacmap(x, dx, t, target_name, niters = 450):
    powers = np.zeros((x.shape[1], len(period_grid)))
    mf = np.zeros(x.shape[1])
    mp = np.zeros(x.shape[1])
    mfap = np.zeros(x.shape[1])
    for i in tqdm(range(x.shape[1])):
        p, m, mmp, f = periodo(i, x.T, dx.T, t)
        powers[i] = p
        mf[i] = m
        mfap[i] = f
        mp[i] = mmp
    PacMap_map = PacMapDisp(powers, mf, mp, mfap, target_name, niters = niters)
    return(PacMap_map, mf, mp, mfap)

def select_per(Per, mxfq):
    U = Per * 0.05
    indlw = []
    for i in range(len(mxfq)):
        if mxfq[i]>Per-U and mxfq[i]<Per+U:
            indlw.append(i)
    return(indlw)

def preselect(PacMap_map, max_freq, max_pow, max_fap, prot):
    Pac_an_1  = PacMap_map[select_per(yr, max_freq)]
    sample_weight_an_1 = max_pow[select_per(yr, max_freq)]
    size_an_1 = max_fap[select_per(yr, max_freq)]
    Pac_an_2  = PacMap_map[select_per(yr/2, max_freq)]
    sample_weight_an_2 = max_pow[select_per(yr/2, max_freq)]
    size_an_2 = max_fap[select_per(yr/2, max_freq)]
    Pac_an_3  = PacMap_map[select_per(yr/3, max_freq)]
    sample_weight_an_3 = max_pow[select_per(yr/3, max_freq)]
    size_an_3 = max_fap[select_per(yr/3, max_freq)]

    Pac_prot_1  = PacMap_map[select_per(prot, max_freq)]
    sample_weight_prot_1 = max_pow[select_per(prot, max_freq)]
    size_prot_1 = max_fap[select_per(prot, max_freq)]
    Pac_prot_2  = PacMap_map[select_per(prot/2, max_freq)]
    sample_weight_prot_2 = max_pow[select_per(prot/2, max_freq)]
    size_prot_2 = max_fap[select_per(prot/2, max_freq)]

    Pac_prot = np.concatenate((Pac_prot_1, Pac_prot_2))
    sample_weight_prot = np.concatenate((sample_weight_prot_1, sample_weight_prot_2))
    size_prot = np.concatenate((size_prot_1, size_prot_2))
    Pac_an   = np.concatenate((Pac_an_1, Pac_an_2, Pac_an_3))
    sample_weight_an = np.concatenate((sample_weight_an_1, sample_weight_an_2, sample_weight_an_3))
    size_an   = np.concatenate((size_an_1, size_an_2, size_an_3))

    if chatty:
        print('         a priori flagged activity lines :', Pac_prot.shape[0])
        print('         a priori flagged telluric lines :', Pac_an.shape[0])

    prot_weight_ratio = Pac_an.shape[0]/Pac_prot.shape[0]

    class_id = np.concatenate((np.zeros_like(Pac_prot.T[0]), np.ones_like(Pac_an.T[0])))
    Pac_split = np.concatenate((Pac_prot, Pac_an))
    sample_weight = np.concatenate((sample_weight_prot*prot_weight_ratio, sample_weight_an))
    size_split = np.concatenate((size_prot*prot_weight_ratio, size_an))
    return(Pac_split, class_id, sample_weight, size_split)

def plot_decision_function(classifier, sample_weight, axis, title, Pac_split, class_id, size_split):
    # plot the decision function
    mx, my = np.max(np.abs(Pac_split[:, 0])), np.max(np.abs(Pac_split[:, 1]))
    xx, yy = np.meshgrid(np.linspace(-mx-2, mx+2, 500), np.linspace(-my-2, my+2, 500))

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot the line, the points, and the nearest vectors to the plane
    axis.contourf(xx, yy, Z, levels=1, alpha=0.5, cmap=plt.cm.plasma)
    axis.scatter(
        Pac_split[:, 0],
        Pac_split[:, 1],
        c=class_id,
        s=10*size_split,
        alpha=0.9,
        cmap=plt.cm.plasma,
        edgecolors="black",
    )

    axis.axis("off")
    axis.set_title(title)

def ind_outliers(v, k):
    madv= wapiti_tools.mad(v)
    rejection = wapiti_tools.absolute_deviation(v) > k*madv
    rejection_index = np.where(rejection)[0]
    return(rejection_index)

def compute_anomaly_degree(pca):
    """
    Compute the anomaly degree of each epoch based on the maximum ratio between the absolute deviation and the median
    absolute deviation among all principal vectors V_n. This is used to identify anomalous observations.

    Parameters:
        pca (WPCA): A fitted wPCA model containing principal components.

    Returns:
        numpy.ndarray: An array of size (n_epochs,) containing the anomaly degree of each epoch.
    """
    # Initialize distance_mad array to store the ratio of absolute deviation to median absolute deviation
    distance_mad = np.zeros((pca.components_.shape[1], pca.components_.shape[0]))

    for idx in range(pca.components_.shape[0]):
        # Calculate MAD of current component
        madv = wapiti_tools.mad(pca.components_[idx])
        # Calculate absolute deviation of current component
        ad = wapiti_tools.absolute_deviation(pca.components_[idx])
        # Store ratio of absolute deviation to median absolute deviation
        distance_mad[:, idx] = ad/madv

    # Get the maximum value of the distance_mad array along the rows
    D = np.max(distance_mad, axis=1)

    return D


def find_optimal_rejection(D, time_binned, rvs_binned, drvs_binned, frequency, n_components, regularization=0, threshold = 10):
    """
    This function computes the false alarm probabilities (FAPs) for each set of RV data after removing epochs one by one,
    sorted in decreasing order of their anomaly degree D.

    Args:
    - D: array of anomaly degree
    - time_binned: array of binned time values
    - rvs_binned: array of binned RV values
    - drvs_binned: array of binned RV uncertainty values
    - frequency: array of angular frequencies at which to compute the Lomb-Scargle periodogram
    - n_components: number of principal components to retain for the WPCA analysis (default=20)

    Returns:
    - faps: array of FAPs for each set of RV data after removing epochs one by one
    """

    # Sort the epochs by decreasing D value
    index_sort = np.argsort(D)[::-1]

    faps = []
    d_idx = 0
    while D[index_sort][d_idx] >= threshold:
        # Remove the epoch with the highest D value
        time_used = np.delete(time_binned, index_sort[:d_idx+1])
        rvs_used = np.copy(rvs_binned)
        drvs_used = np.copy(drvs_binned)
        rvs_used = np.delete(rvs_used, index_sort[:d_idx+1], axis=0)
        drvs_used = np.delete(drvs_used , index_sort[:d_idx+1], axis=0)
        rvs_used = rvs_used.T
        drvs_used = drvs_used.T

        # Compute the weighted average and variance of the remaining RV data
        ma_rvs = np.ma.MaskedArray((rvs_used), mask=np.isnan((rvs_used)))
        ma_drvs = np.ma.MaskedArray((drvs_used), mask=np.isnan((drvs_used)))
        average = np.ma.average(ma_rvs, weights=1/ma_drvs**2, axis=1)
        variance = np.ma.average((ma_rvs-average.reshape(-1, 1))**2, weights=1/ma_drvs**2, axis=1)
        mean_X = average.data.reshape(-1, 1)
        std_X = np.sqrt(variance.data.reshape(-1, 1))

        # Normalize the RVs and RV uncertainties
        rv = (np.copy(rvs_used)-mean_X)/std_X
        drv = np.copy(drvs_used)/std_X

        # Compute weights for the RVs based on the normalized RV uncertainties
        weights = 1. / drv
        weights[np.isnan(rv)] = 0

        # Fit a WPCA model to the normalized RVs and weights
        pca_0 = WPCA(n_components=n_components)
        pca_0.regularization = regularization
        pca_0.fit(rv, weights=weights)
        wpca_model = pca_0.reconstruct(rv, weights=weights)

        rv_0, std_rv_0 = [], []
        for idx in tqdm(range(len(time_used)), leave=False):
            rv_temp, std_rv_temp = odd_ratio_mean(rvs_used.T[idx], drvs_used.T[idx])
            rv_0.append(rv_temp)
            std_rv_0.append(std_rv_temp)
        rv_0, std_rv_0 = np.array(rv_0), np.array(std_rv_0)

        average, std = odd_ratio_mean(rv_0, std_rv_0)

        wpca_model = pca_0.reconstruct([(rv_0 - average)/std], weights=[std/std_rv_0])

        rv_wapiti = (((rv_0 - average)/std - wpca_model)*std + average)[0]
        std_rv_wapiti = std_rv_0

        # LombScargle
        ls = LombScargle(time_used, rv_wapiti, std_rv_wapiti)
        power = ls.power(frequency)
        fap = ls.false_alarm_probability(power.max())
        faps.append(fap)

        d_idx += 1

    faps = np.array(faps)

    return faps

def odd_ratio_mean(value, err, odd_ratio=1e-4, nmax=10):
    # Vectorized implementation of odd_ratio_mean

    # Mask NaNs
    mask = np.isfinite(value) & np.isfinite(err)
    if not np.any(mask):
        return np.nan, np.nan

    # Apply mask
    value = value[mask]
    err = err[mask]

    # Initial guess
    guess = np.nanmedian(value)

    for nite in range(nmax):
        nsig = (value - guess) / err
        gg = np.exp(-0.5 * nsig**2)
        odd_bad = odd_ratio / (gg + odd_ratio)
        odd_good = 1 - odd_bad
        w = odd_good / err**2
        guess = np.nansum(value * w) / np.nansum(w)

    bulk_error = np.sqrt(1 / np.nansum(odd_good / err**2))

    return guess, bulk_error

## Main loop

def main_loop(target, template, prot):
    if chatty:
        print('################################################')
        print('entering in ' + target + ' s loop ')
        print('     Loading data')
    # load
    times_lbl, d2vs_all, dd2vs_all, snrs, berv = get_data(target, template)

    if chatty:
        print('     Night binning')
    # night binn
    d2vs_binned = []
    dd2vs_binned = []
    for idx in tqdm(range(d2vs_all.shape[1])):
        time_binned, d2v, dd2v = wapiti_tools.night_bin(times_lbl, d2vs_all[:, idx], dd2vs_all[:, idx])
        d2vs_binned.append(d2v)
        dd2vs_binned.append(dd2v)
    d2vs_binned = np.array(d2vs_binned).T
    dd2vs_binned = np.array(dd2vs_binned).T
    time_binned -= 2450000
    _, snrs = wapiti_tools.night_bin(times_lbl, snrs)
    _, berv_used = wapiti_tools.night_bin(times_lbl, berv)
    if chatty:
        print('         Mean SNR: ', np.mean(snrs))
        print('         BERV max: ', np.min(berv_used), np.max(berv_used))

    if chatty:
        print('     remove nans')
        print('         Number of lines before rejection: ', d2vs_binned.shape[1])
    # remove Nan
    valid = wapiti_tools.compute_valid_lines(d2vs_binned.T, time_binned, 0.5)
    if chatty:
        print('         Number of lines after rejection: ', d2vs_binned[:, valid].shape[1])

    time_used = np.copy(time_binned)
    d2vs_used = np.copy(d2vs_binned)[:, valid]
    dd2vs_used = np.copy(dd2vs_binned)[:, valid]

    if chatty:
        print('     not filtered PCA')
    # run a first PCA without anyfiltering

    pca_d2v = run_PCA(d2vs_used, dd2vs_used)
    outliers = ind_outliers(pca_d2v.components_[0], initial_mad_rejection)
    outloop = 0
    while len(outliers)>=1 and outloop < looplier :
        if chatty:
            print("         Outlier spotted... Removing ")
        D = compute_anomaly_degree(pca_d2v)
        faps = find_optimal_rejection(D, time_used, d2vs_used, dd2vs_used, frequency, n_components = n_component_rejection, threshold = initial_mad_rejection-(outloop/2))
        if chatty:
            print(f'The signal is the most significant when removing the {np.argmin(faps)+1} observations of highest D with a log fap of {np.log10(np.min(faps)):.2f}')


        index_sort = np.argsort(D)[::-1]
        optimal_indx = np.argmin(faps)
        time_used = np.delete(time_used, index_sort[:optimal_indx+1])
        berv_used = np.delete(berv_used, index_sort[:optimal_indx+1])
        # Mask the copies of the arrays using the mask
        d2vs_used = np.delete(np.copy(d2vs_used), index_sort[:optimal_indx+1], axis=0)
        dd2vs_used = np.delete(np.copy(dd2vs_used) , index_sort[:optimal_indx+1], axis=0)
        pca_d2v = run_PCA(d2vs_used, dd2vs_used)

        outloop += 1
        outliers = ind_outliers(pca_d2v.components_[0], initial_mad_rejection-(outloop/2))

    # two step filtering method
        # step 1 pacmap

    if chatty:
        f, p = LombScargle(time_used, pca_d2v.components_[0]).autopower(minimum_frequency=0.0005, maximum_frequency=1/1.5) #nyquist_factor=15)7
        ls = LombScargle(time_used, pca_d2v.components_[0])
        Prow = p[np.argmin(np.abs(f - 1/prot))]
        print('         Before filtering W1 Prot fap ', np.log(ls.false_alarm_probability(Prow)))
        print('     run PacMap')
    PacMap_map, max_f, max_p, max_fp = run_pacmap(d2vs_used, dd2vs_used, time_used, target)

    if chatty:
        print('     run SVM')
        # step 2 SVM
            # define a priori the two classes
    PacMap_split, class_id, sample_weight, size_split = preselect(PacMap_map, max_f, max_p, max_fp, prot)

            #run SVM
    # This model takes into account some dedicated sample weights.
    clf_weights = svm.SVC(kernel='rbf', gamma='auto')
    clf_weights.fit(PacMap_split, class_id, sample_weight=sample_weight)
    predicted_class = clf_weights.predict(PacMap_map)

    if plotty:
        fig, axes = plt.subplots( 1, 1, figsize=(16, 10))
        axes.scatter(PacMap_map[:,0], PacMap_map[:,1], c='k', alpha = 0.05)
        plot_decision_function(clf_weights, sample_weight, axes, target, PacMap_split, class_id, size_split)
        plt.savefig(cwd+"/out/" + target + "SVM.pdf", format="pdf", bbox_inches="tight")
        #plt.ion()

    d2v_filtr  = d2vs_used.T[predicted_class == 0].T
    sd2v_filtr = dd2vs_used.T[predicted_class == 0].T
    if chatty:
        print('         SVM Activity line Ratio = ', d2v_filtr.shape[1]/d2vs_used.shape[1])
        print('         # line filtr = ', d2v_filtr.shape[1])
        print('     filtered PCA')
    # run the PCA on the filtered lines
    pca_filtr = run_PCA(d2v_filtr, sd2v_filtr)
    if plotty:
        fig, ax = plt.subplots(1, 3, figsize=(20, 5))
        ax[0].plot(time_used, pca_d2v.components_[0], 'ro', alpha=0.7)
        ax[0].plot(time_used, pca_filtr.components_[0], 'bo')
        ax[0].set_ylabel(f'W {1}', size=12, weight='bold')
        # Set the x-axis label
        ax[0].set_xlabel('Time [BJD]', size=12, weight='bold')
        f, p = LombScargle(time_used, pca_filtr.components_[0]).autopower(minimum_frequency=1/2000, maximum_frequency=1/1.5) #nyquist_factor=15)
        ax[1].plot(1/f, p, 'b', label='Filtered')
        f, p = LombScargle(time_used, pca_d2v.components_[0]).autopower(minimum_frequency=1/2000, maximum_frequency=1/1.5) #nyquist_factor=15)
        ax[1].plot(1/f, p, 'r', alpha=0.7, label='Not filtered')
        ax[1].set_ylabel("power")
        ax[1].set_xscale('log')
        ls = LombScargle(time_used, pca_filtr.components_[0])
        fap = ls.false_alarm_level(0.1)
        ax[1].axhline(fap, linestyle='-', color='k')
        fap = ls.false_alarm_level(0.01)
        ax[1].axhline(fap, linestyle='--', color='k')
        fap = ls.false_alarm_level(0.001)
        ax[1].axhline(fap, linestyle=':', color='k')
        ax[1].axvline(prot, linestyle=':', alpha=0.5)
        ax[1].set_xlabel('Period [d]', size=12, weight='bold')
        ax[1].set_ylabel('Power', size=12, weight='bold')
        ax[1].legend()
        ax[2].plot(berv_used, pca_d2v.components_[0], 'ro', alpha=0.7)
        ax[2].plot(berv_used, pca_filtr.components_[0], 'bo')
        ax[2].set_ylabel(f'W {1}', size=12, weight='bold')
        # Set the x-axis label
        ax[2].set_xlabel('BERV [km/s]', size=12, weight='bold')
        # Set the font size of the tick labels
        ax[2].tick_params(labelsize=12)
        plt.savefig(cwd+"/out/" + target + "W1.pdf", format="pdf", bbox_inches="tight")
        #plt.ion()
    if chatty:
        f, p = LombScargle(time_used, pca_filtr.components_[0]).autopower(minimum_frequency=1/2000, maximum_frequency=1/1.5)
        ls = LombScargle(time_used, pca_filtr.components_[0])
        Prow = p[np.argmin(np.abs(f - 1/prot))]
        print('         After filtering W1 Prot fap ', np.log(ls.false_alarm_probability(Prow)))

        print('exiting ' + target + ' s loop')

    return(time_used, pca_filtr)


## Main


for i in range(len(Target)):
    try :
        time, pca_after = main_loop(Target[i], Template[i], Prot[i])
        np.save(cwd+"/out/" + Target[i] + "W1.npy", np.array([time, pca_after.components_[0]]))
        gc.collect()
    except:
        pass










