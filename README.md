# dLWPCA

Here is my code to reduce differential line width (dLW) data from lbl using wPCA(https://github.com/jakevdp/wpca) and then analyze it using GPr


This repository is organized this way :

There is 3 different methods obtain a clear, and BERV filtered, _W1_ signal:

  - `PCAul.py` is about preparing data for the wCPA step. It takes tellurics corrected lbl data, filters some the NaNs, binn it by night remove some outliers and Normalize data. The final data are saved into a Table. and then we run the wPCA process over the prepared data.
  - `MerwPCAn.ipynb` is about preparing data for the wCPA step using _wapiti_ tools (https://github.com/HkmMerwan/wapiti), and extract the activity signal by looking at the PCA component were the activity appears and select only the lines that best fit the this component.
  - `PCAeriodogram.py` extract the activity signal by looking only at the lines were the activity appears in their _dLW_ periodogram.
  - `SPCAndie.py` Use machine learning tools such as tSNE, Umap, TriMap or PacMap to sort the lines given their dLW periodogram.
  - `Bcorr.py` Sort the lines given their correlation regarding the magnetic field.
  
And one to analyze the resulting _W1_:
  - `GwPCA.py` is about to analyze the first component return by the PCA by applying a Gaussian Processes regresion on it and compares it to available data on the magnetic field and radial velocities. The GP used is a custom tedi GP (https://github.com/jdavidrcamacho/tedi)
