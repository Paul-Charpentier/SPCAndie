# dLWPCA

Here is my code to reduce data from lbl using wPCA(https://github.com/jakevdp/wpca) and then analyze it using GPr

The data here are the dLW data from the star Au Mic. 

This repository is organized this way : 

  - `dLWCPA.py` is about preparing data for the wCPA step. It takes tellurics corrected lbl data, filters some the NaNs, binn it by night remove some outliers and Normalize data. The final data are saved into a Table. and then we run the wPCA process over the prepared data.
  - `GwPCA.py` is about to analyze the 2 first component of the resulting wPCA. In particular it uses a Gaussian Processes regresion to do so. And cross correlate it to magnetic field data from ZeeTurbo. 

Note that there is dependencies on custom tedi GP (https://github.com/jdavidrcamacho/tedi)
