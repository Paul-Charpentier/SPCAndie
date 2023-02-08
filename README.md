# dLWPCA

Here is my code to reduce data from lbl using wPCA(https://github.com/jakevdp/wpca) and then analyze it using GPr

The data here are the dLW data from the star Au Mic. 

This repository is organized this way : 

  - `BuildTablesTowPCA.py` is about preparing data for the wCPA step. It takes tellurics corrected lbl data, binn it by night and filters some the NaNs and. The final data are saved into a Table.
  - `dLWCPA.py` Run the wPCA process over the prepared data.
  - `GwPCA.py` is about to analyze the 2 first component of the resulting wPCA. In particular it uses a Gaussian Processes regresion to do so. 


Note that there is dependencies on my owm custom tedi GP (https://github.com/Paul-Charpentier/tediGP)
