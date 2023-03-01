# dLWPCA

Here is my code to reduce differential line width (dLW) data from lbl using wPCA(https://github.com/jakevdp/wpca) and then analyze it using GPr


This repository is organized this way : 

  - `dLWCPA.py` is about preparing data for the wCPA step. It takes tellurics corrected lbl data, filters some the NaNs, binn it by night remove some outliers and Normalize data. The final data are saved into a Table. and then we run the wPCA process over the prepared data.
  - `GwPCA.py` is about to analyze the first component return by the PCA by applying a Gaussian Processes regresion on it and compares it to available data on the magnetic field and radial velocities.

the GP used is a custom tedi GP (https://github.com/jdavidrcamacho/tedi)
